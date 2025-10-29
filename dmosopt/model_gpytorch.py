from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import gc
import numpy as np
from scipy.cluster.vq import kmeans2
from dmosopt.MOEA import top_k_MO, filter_samples

try:
    pass
except ImportError:
    _has_pykeops = False
else:
    _has_pykeops = True


def handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.
    The goal is to avoid division by very small or zero values.
    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.
    Typically for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    Code from sklearn.preprocessing.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, np.ndarray):
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[constant_mask] = 1.0
        return scale


try:
    import torch
    import gpytorch
    from contextlib import ExitStack

    def find_best_gpu_setting(
        nInput,
        nOutput,
        train_f,
        train_x,
        train_y,
        n_devices,
        preconditioner_size,
        logger=None,
    ):
        N = train_x.size(0)

        # Find the optimum partition/checkpoint size by decreasing in powers of 2
        # Start with no partitioning (size = 0)
        settings = [0] + [
            int(n) for n in np.ceil(N / 2 ** np.arange(1, np.floor(np.log2(N))))
        ]

        for checkpoint_size in settings:
            logger.info(
                "gpytorch: Number of devices: {} -- Kernel partition size: {}".format(
                    n_devices, checkpoint_size
                )
            )
            try:
                # Try a full forward and backward pass with this setting to check memory usage
                _ = train_f(
                    nInput,
                    nOutput,
                    train_x,
                    train_y,
                    n_iter=1,
                    checkpoint_size=checkpoint_size,
                    preconditioner_size=preconditioner_size,
                )

                # when successful, break out of for-loop and jump to finally block
                break
            except RuntimeError as e:
                logger.error("RuntimeError: {}".format(e))
            except AttributeError as e:
                logger.error("AttributeError: {}".format(e))
            finally:
                # handle CUDA OOM error
                gc.collect()
                torch.cuda.empty_cache()

        return checkpoint_size

    class GPyTorchDSPPMaternLayer(gpytorch.models.deep_gps.dspp.DSPPLayer):
        def __init__(
            self,
            input_dims,
            output_dims,
            num_inducing_points=128,
            inducing_points=None,
            linear_mean=True,
            ard_num_dims=None,
            lengthscale_bounds=None,
            use_cuda=False,
            n_devices=1,
        ):
            if output_dims is None:
                if inducing_points is None:
                    inducing_points = torch.randn(num_inducing_points, input_dims)
                batch_shape = torch.Size([])
            else:
                if inducing_points is None:
                    inducing_points = torch.randn(
                        output_dims, num_inducing_points, input_dims
                    )
                batch_shape = torch.Size([output_dims])

            if use_cuda:
                inducing_points = inducing_points.cuda()

            batch_shape = torch.Size([output_dims])

            lengthscale_constraint = None
            if lengthscale_bounds is not None:
                lengthscale_constraint = gpytorch.constraints.Interval(
                    lengthscale_bounds[0], lengthscale_bounds[1]
                )

            variational_distribution = (
                gpytorch.variational.CholeskyVariationalDistribution(
                    num_inducing_points=num_inducing_points, batch_shape=batch_shape
                )
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )

            super().__init__(variational_strategy, input_dims, output_dims)
            self.mean_module = (
                gpytorch.means.LinearMean(input_dims)
                if linear_mean
                else gpytorch.means.ConstantMean()
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                (
                    gpytorch.kernels.MaternKernel(
                        nu=2.5,
                        batch_shape=batch_shape,
                        ard_num_dims=ard_num_dims,
                        lengthscale_constraint=lengthscale_constraint,
                    )
                    if not _has_pykeops
                    else gpytorch.kernels.keops.MaternKernel(
                        nu=2.5,
                        batch_shape=batch_shape,
                        ard_num_dims=ard_num_dims,
                        lengthscale_constraint=lengthscale_constraint,
                    )
                ),
                batch_shape=batch_shape,
                ard_num_dims=None,
            )

            if n_devices is not None and n_devices > 1:
                self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                    self.covar_module, device_ids=range(n_devices)
                )

        def forward(self, x, mean_input=None, **kwargs):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class GPyTorchMultitaskDSPPMatern(gpytorch.models.deep_gps.dspp.DSPP):
        def __init__(
            self,
            train_x,
            likelihood,
            num_tasks,
            num_hidden_dims,
            Q=8,
            num_inducing_points=128,
            lengthscale_bounds=None,
            batch_size=None,
            use_cuda=False,
            n_devices=1,
        ):
            super().__init__(Q)

            self.num_tasks = num_tasks
            self.batch_size = batch_size
            self.likelihood = likelihood
            self.use_cuda = use_cuda

            # Use K-means to initialize inducing points (only helpful for the first layer)
            if num_inducing_points > train_x.shape[0]:
                num_inducing_points = train_x.shape[0]
            inducing_points = train_x[
                torch.randperm(train_x.shape[0])[0:num_inducing_points]
            ]
            inducing_points = inducing_points.clone().data.cpu().numpy()
            inducing_points = torch.tensor(
                kmeans2(train_x.data.cpu().numpy(), inducing_points, minit="matrix")[0]
            )

            if self.use_cuda:
                inducing_points = inducing_points.cuda()

            hidden_layer = GPyTorchDSPPMaternLayer(
                input_dims=train_x.shape[-1],
                output_dims=num_hidden_dims,
                lengthscale_bounds=lengthscale_bounds,
                num_inducing_points=num_inducing_points,
                inducing_points=inducing_points,
                linear_mean=True,
                use_cuda=self.use_cuda,
                n_devices=n_devices,
            )
            last_layer = GPyTorchDSPPMaternLayer(
                input_dims=hidden_layer.output_dims,
                output_dims=num_tasks,
                lengthscale_bounds=lengthscale_bounds,
                num_inducing_points=num_inducing_points,
                inducing_points=None,
                linear_mean=False,
                use_cuda=self.use_cuda,
                n_devices=n_devices,
            )

            self.hidden_layer = hidden_layer
            self.last_layer = last_layer

        def forward(self, inputs, **kwargs):
            hidden_rep1 = self.hidden_layer(inputs, **kwargs)
            output = self.last_layer(hidden_rep1, **kwargs)
            return output

        def predict(self, xin):
            from torch.utils.data import DataLoader

            batch_size = self.batch_size
            if self.batch_size is None:
                batch_size = xin.shape[0]

            x = torch.from_numpy(xin)

            in_loader = DataLoader(x, batch_size=batch_size, shuffle=False)

            self.eval()
            self.likelihood.eval()
            with (
                gpytorch.settings.fast_computations(log_prob=False, solves=False),
                torch.no_grad(),
            ):
                means, variances = [], []
                for x_batch in in_loader:
                    if self.use_cuda:
                        x_batch = x_batch.cuda()
                    batch_preds = self.likelihood(self(x_batch, mean_input=x_batch))
                    batch_mean = batch_preds.mean.mean(0)
                    batch_var = batch_preds.variance.mean(0)

                    means.append(batch_mean)
                    variances.append(batch_var)

            return torch.cat(means), torch.cat(variances)

    class GPyTorchDGPMaternLayer(gpytorch.models.deep_gps.DeepGPLayer):
        def __init__(
            self,
            input_dims,
            output_dims=None,
            num_inducing_points=128,
            inducing_points=None,
            linear_mean=True,
            ard_num_dims=None,
            lengthscale_bounds=None,
            use_cuda=False,
            n_devices=1,
        ):
            if output_dims is None:
                if inducing_points is None:
                    inducing_points = torch.randn(num_inducing_points, input_dims)
                batch_shape = torch.Size([])
            else:
                if inducing_points is None:
                    inducing_points = torch.randn(
                        output_dims, num_inducing_points, input_dims
                    )
                batch_shape = torch.Size([output_dims])

            if use_cuda:
                inducing_points = inducing_points.cuda()

            lengthscale_constraint = None
            if lengthscale_bounds is not None:
                lengthscale_constraint = gpytorch.constraints.Interval(
                    lengthscale_bounds[0], lengthscale_bounds[1]
                )

            variational_distribution = (
                gpytorch.variational.CholeskyVariationalDistribution(
                    num_inducing_points=num_inducing_points, batch_shape=batch_shape
                )
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )

            super().__init__(variational_strategy, input_dims, output_dims)
            self.mean_module = (
                gpytorch.means.LinearMean(input_dims)
                if linear_mean
                else gpytorch.means.ConstantMean()
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                (
                    gpytorch.kernels.MaternKernel(
                        nu=2.5,
                        batch_shape=batch_shape,
                        ard_num_dims=ard_num_dims,
                        lengthscale_constraint=lengthscale_constraint,
                    )
                    if not _has_pykeops
                    else gpytorch.kernels.keops.MaternKernel(
                        nu=2.5,
                        batch_shape=batch_shape,
                        ard_num_dims=ard_num_dims,
                        lengthscale_constraint=lengthscale_constraint,
                    )
                ),
                batch_shape=batch_shape,
                ard_num_dims=None,
            )
            if n_devices is not None and n_devices > 1:
                self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                    self.covar_module, device_ids=range(n_devices)
                )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    class GPyTorchMultitaskDeepGPMatern(gpytorch.models.deep_gps.DeepGP):
        def __init__(
            self,
            train_x,
            likelihood,
            num_tasks,
            num_hidden_dims,
            num_inducing_points=128,
            lengthscale_bounds=None,
            batch_size=None,
            use_cuda=False,
            n_devices=1,
        ):
            super().__init__()

            self.num_tasks = num_tasks
            self.batch_size = batch_size
            self.use_cuda = use_cuda

            # We're going to use a multitask likelihood instead of the standard GaussianLikelihood
            self.likelihood = likelihood

            # Use K-means to initialize inducing points (only helpful for the first layer)
            if num_inducing_points > train_x.shape[0]:
                num_inducing_points = train_x.shape[0]
            inducing_points = train_x[
                torch.randperm(train_x.shape[0])[0:num_inducing_points]
            ]
            inducing_points = inducing_points.clone().data.cpu().numpy()
            inducing_points = torch.tensor(
                kmeans2(train_x.data.cpu().numpy(), inducing_points, minit="matrix")[0]
            )

            self.hidden_layer = GPyTorchDGPMaternLayer(
                input_dims=train_x.shape[-1],
                output_dims=num_hidden_dims,
                lengthscale_bounds=lengthscale_bounds,
                num_inducing_points=num_inducing_points,
                inducing_points=inducing_points,
                linear_mean=True,
                use_cuda=use_cuda,
                n_devices=n_devices,
            )
            self.last_layer = GPyTorchDGPMaternLayer(
                input_dims=self.hidden_layer.output_dims,
                output_dims=num_tasks,
                lengthscale_bounds=lengthscale_bounds,
                num_inducing_points=num_inducing_points,
                inducing_points=None,
                linear_mean=False,
                use_cuda=use_cuda,
                n_devices=n_devices,
            )

        def forward(self, inputs):
            hidden_rep1 = self.hidden_layer(inputs)
            output = self.last_layer(hidden_rep1)
            return output

        def predict(self, xin):
            from torch.utils.data import DataLoader

            batch_size = self.batch_size
            if self.batch_size is None:
                batch_size = xin.shape[0]

            x = torch.from_numpy(xin)
            in_loader = DataLoader(x, batch_size=batch_size, shuffle=False)

            self.eval()
            self.likelihood.eval()

            with (
                gpytorch.settings.fast_computations(log_prob=False, solves=False),
                torch.no_grad(),
            ):
                means, variances = [], []
                for x_batch in in_loader:
                    if self.use_cuda:
                        x_batch = x_batch.cuda()
                    # The output of the model is a multitask MVN, where both the data points
                    # and the tasks are jointly distributed
                    # To compute the marginal predictive NLL of each data point,
                    # we will call `to_data_independent_dist`,
                    # which removes the data cross-covariance terms from the distribution.
                    batch_preds = self.likelihood(
                        self(x_batch)
                    ).to_data_independent_dist()
                    batch_mean = batch_preds.mean.mean(0)
                    batch_var = batch_preds.variance.mean(0)

                    means.append(batch_mean)
                    variances.append(batch_var)

            return torch.cat(means), torch.cat(variances)

    class GPyTorchExactGPModelMatern(gpytorch.models.ExactGP):
        def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            ard_num_dims=None,
            lengthscale_bounds=None,
            linear_mean=True,
            batch_size=None,
            n_devices=1,
        ):
            super().__init__(train_x, train_y, likelihood)
            lengthscale_constraint = None
            if lengthscale_bounds is not None:
                lengthscale_constraint = gpytorch.constraints.Interval(
                    lengthscale_bounds[0], lengthscale_bounds[1]
                )
            self.batch_size = batch_size
            batch_shape = torch.Size()
            if batch_size is not None:
                batch_shape = torch.Size([batch_size])
            input_dims = train_x.shape[1]
            self.mean_module = (
                gpytorch.means.LinearMean(input_dims, batch_shape=batch_shape)
                if linear_mean
                else gpytorch.means.ConstantMean(batch_shape=batch_shape)
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    ard_num_dims=ard_num_dims,
                    lengthscale_constraint=lengthscale_constraint,
                    batch_shape=batch_shape,
                ),
                batch_shape=batch_shape,
            )
            if n_devices is not None and n_devices > 1:
                self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                    self.covar_module, device_ids=range(n_devices)
                )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            mvn = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            if self.batch_size is not None:
                mmvn = (
                    gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                        mvn
                    )
                )
                return mmvn
            else:
                return mvn

    class GPyTorchMultitaskExactGPModelMatern(gpytorch.models.ExactGP):
        def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            num_tasks,
            rank=1,
            ard_num_dims=None,
            lengthscale_bounds=None,
            linear_mean=True,
            batch_size=None,
            n_devices=1,
        ):
            super().__init__(train_x, train_y, likelihood)
            input_dims = train_x.shape[1]
            self.num_tasks = num_tasks
            self.batch_size = batch_size
            batch_shape = torch.Size()
            if batch_size is not None:
                batch_shape = torch.Size([batch_size])
            lengthscale_constraint = None
            if lengthscale_bounds is not None:
                lengthscale_constraint = gpytorch.constraints.Interval(
                    lengthscale_bounds[0], lengthscale_bounds[1]
                )
            self.mean_module = gpytorch.means.MultitaskMean(
                (
                    gpytorch.means.LinearMean(input_dims, batch_shape=batch_shape)
                    if linear_mean
                    else gpytorch.means.ConstantMean(batch_shape=batch_shape)
                ),
                num_tasks=num_tasks,
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.MaternKernel(
                    ard_num_dims=ard_num_dims,
                    lengthscale_constraint=lengthscale_constraint,
                    batch_shape=batch_shape,
                ),
                batch_shape=batch_shape,
                num_tasks=num_tasks,
                rank=rank,
            )
            if n_devices is not None and n_devices > 1:
                self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                    self.covar_module, device_ids=range(n_devices)
                )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            mvn = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
            if self.batch_size is not None:
                mmvn = (
                    gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                        mvn, task_dim=-1
                    )
                )
                return mmvn
            else:
                return mvn

except ImportError:
    _has_gpytorch = False
else:
    _has_gpytorch = True


class ModelType(Enum):
    """Model type enumeration for adaptive training strategies"""

    EXACT_GP = "exact_gp"
    VARIATIONAL_GP = "variational_gp"
    DEEP_GP = "deep_gp"
    DEEP_STOCHASTIC = "deep_stochastic"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping criteria"""

    min_iterations: int = 1000  # Minimum iterations before checking
    window_size: int = 500  # Size of moving window for statistics
    threshold_pct: float = 0.1  # Percentage change threshold
    patience: int = 3  # Number of checks before stopping
    warmup_iterations: int = 100  # Iterations before activating checks
    relative_tolerance: float = 1e-2  # Relative tolerance for loss comparison
    absolute_tolerance: float = 1e-3  # Absolute tolerance for loss comparison

    @classmethod
    def for_model_type(cls, model_type: ModelType) -> "EarlyStoppingConfig":
        """Factory method for model-specific configurations"""
        configs = {
            ModelType.EXACT_GP: cls(
                min_iterations=1000,
                window_size=200,
                threshold_pct=0.01,  # Tighter for exact GP
                patience=2,
                warmup_iterations=50,
            ),
            ModelType.VARIATIONAL_GP: cls(
                min_iterations=1000,
                window_size=500,
                threshold_pct=0.5,
                patience=3,
                warmup_iterations=200,
            ),
            ModelType.DEEP_GP: cls(
                min_iterations=1500,
                window_size=500,
                threshold_pct=1.0,
                patience=3,
                warmup_iterations=200,
            ),
            ModelType.DEEP_STOCHASTIC: cls(
                min_iterations=2000,
                window_size=500,
                threshold_pct=1.0,
                patience=3,
                warmup_iterations=200,
            ),
        }
        return configs.get(model_type, cls())


class AdaptiveEarlyStopping:
    """
    Early stopping with multiple convergence criteria.
    """

    def __init__(self, config: EarlyStoppingConfig, logger=None):
        self.config = config
        self.consecutive_stops = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.logger = logger

    def should_stop(
        self,
        iteration: int,
        loss_history: np.ndarray,
        compute_validation: Optional[Callable[[], float]] = None,
    ) -> Tuple[bool, str]:
        """
        Determine if training should stop based on multiple criteria.

        Args:
            iteration: Current iteration number
            loss_history: Array of loss values
            compute_validation: Optional function to compute validation loss

        Returns:
            (should_stop, reason): Boolean and explanation string
        """

        # Percentage change in moving window
        stop_pct, reason_pct = self._check_percentage_change(loss_history)

        # Absolute convergence
        stop_abs, reason_abs = self._check_absolute_convergence(loss_history)

        # Relative convergence
        stop_rel, reason_rel = self._check_relative_convergence(loss_history)

        # Loss plateau detection
        stop_plateau, reason_plateau = self._check_plateau(loss_history)

        # Validation loss (if available)
        stop_val, reason_val = (False, "")
        if compute_validation is not None:
            stop_val, reason_val = self._check_validation_loss(
                compute_validation, loss_history
            )

        if iteration < self.config.min_iterations:
            return False, ""

        # Combine criteria with patience mechanism
        criteria_met = sum([stop_pct, stop_abs, stop_rel, stop_plateau, stop_val])

        if criteria_met >= 2:  # At least 2 criteria must agree
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                reasons = [
                    r
                    for r in [
                        reason_pct,
                        reason_abs,
                        reason_rel,
                        reason_plateau,
                        reason_val,
                    ]
                    if r
                ]
                return True, "; ".join(reasons)
        else:
            self.patience_counter = 0

        return False, ""

    def _check_percentage_change(self, loss_history: np.ndarray) -> Tuple[bool, str]:
        """Check if percentage change in moving window is below threshold"""
        if len(loss_history) < self.config.window_size + 1:
            return False, ""

        window = loss_history[-self.config.window_size :]

        # Robust percentage change calculation
        loss_changes = np.diff(window)

        # Avoid division by near-zero values
        denominator = np.maximum(np.abs(window[:-1]), self.config.absolute_tolerance)
        pct_changes = np.abs(loss_changes / denominator) * 100

        mean_pct_change = np.mean(pct_changes)

        if self.logger is not None:
            self.logger.info(f"mean_pct_change: {mean_pct_change}")

        if mean_pct_change < self.config.threshold_pct:
            return True, f"Mean % change ({mean_pct_change:.4f}%) < threshold"
        return False, ""

    def _check_absolute_convergence(self, loss_history: np.ndarray) -> Tuple[bool, str]:
        """Check if absolute change is negligible"""
        if len(loss_history) < self.config.window_size:
            return False, ""

        window = loss_history[-self.config.window_size :]
        max_abs_change = np.max(np.abs(np.diff(window)))
        if self.logger is not None:
            self.logger.info(f"max_abs_change: {max_abs_change}")

        if max_abs_change < self.config.absolute_tolerance:
            return True, f"Max absolute change ({max_abs_change:.2e}) converged"
        return False, ""

    def _check_relative_convergence(self, loss_history: np.ndarray) -> Tuple[bool, str]:
        """Check if relative improvement is negligible"""
        if len(loss_history) < self.config.window_size:
            return False, ""

        window = loss_history[-self.config.window_size :]
        initial = window[0]
        final = window[-1]

        # Avoid division by zero
        if abs(initial) < self.config.absolute_tolerance:
            return False, ""

        relative_change = abs((final - initial) / initial)

        if self.logger is not None:
            self.logger.info(f"relative_change: {relative_change}")

        if relative_change < self.config.relative_tolerance:
            return True, f"Relative change ({relative_change:.2e}) converged"
        return False, ""

    def _check_plateau(self, loss_history: np.ndarray) -> Tuple[bool, str]:
        """Detect if loss has plateaued using statistical test"""
        if len(loss_history) < self.config.window_size * 2:
            return False, ""

        # Split window into two halves
        mid = len(loss_history) - self.config.window_size
        first_half = loss_history[mid : mid + self.config.window_size // 2]
        second_half = loss_history[-self.config.window_size // 2 :]

        # Compare means with small tolerance
        mean_diff = abs(np.mean(first_half) - np.mean(second_half))
        mean_value = np.mean(loss_history[-self.config.window_size :])

        # Check if difference is negligible relative to mean
        relative_diff = mean_diff / (abs(mean_value) + self.config.absolute_tolerance)

        if self.logger is not None:
            self.logger.info(f"relative_diff: {relative_diff}")

        if relative_diff < self.config.relative_tolerance * 2:  # Slightly looser
            return (
                True,
                f"Loss plateau detected (relative difference: {relative_diff:.2e})",
            )
        return False, ""

    def _check_validation_loss(
        self, compute_validation: Callable[[], float], loss_history: np.ndarray
    ) -> Tuple[bool, str]:
        """Check if validation loss has stopped improving"""
        try:
            val_loss = compute_validation()

            if val_loss < self.best_loss - self.config.absolute_tolerance:
                self.best_loss = val_loss
                return False, ""

            # No improvement
            return True, f"No validation improvement (best: {self.best_loss:.4f})"
        except Exception:
            # If validation fails, don't use it as stopping criterion
            return False, ""


def create_adaptive_training_loop(
    model_type: ModelType, n_iter: int = 5000, logger=None
) -> Callable:
    """
    Factory function to create model-specific training loop.

    Returns a closure that encapsulates the training logic with
    adaptive early stopping.
    """
    config = EarlyStoppingConfig.for_model_type(model_type)
    early_stopping = AdaptiveEarlyStopping(config)

    def training_loop(
        optimizer,
        mll,
        gp_model,
        train_x,
        train_y,
        validation_fn: Optional[Callable[[], float]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Execute training loop with adaptive early stopping.

        Args:
            optimizer: PyTorch optimizer
            mll: Marginal log likelihood
            gp_model: GPyTorch model
            train_x: Training inputs
            train_y: Training targets
            validation_fn: Optional validation loss computation

        Returns:
            loss_history: Array of training losses
            metadata: Dict with training statistics
        """
        loss_history = []

        for it in range(n_iter):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = gp_model(train_x)

            # Compute loss
            loss = -mll(output, train_y)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            # Record loss
            loss_value = loss.item()
            loss_history.append(loss_value)

            # Logging
            if it % 100 == 0 and logger is not None:
                logger.info(
                    f"{model_type.value}: iter {it}/{n_iter} - Loss: {loss_value:.3f}"
                )

            # Check early stopping
            if it >= config.warmup_iterations:
                should_stop, reason = early_stopping.should_stop(
                    it, np.array(loss_history), validation_fn
                )

                if should_stop:
                    if logger is not None:
                        logger.info(
                            f"{model_type.value}: early stop at iter {it + 1}: {reason}"
                        )
                    break

        # Prepare metadata
        metadata = {
            "final_iteration": it + 1,
            "final_loss": loss_history[-1] if loss_history else None,
            "converged": should_stop if it >= config.warmup_iterations else False,
            "stop_reason": reason if it >= config.warmup_iterations else "max_iter",
        }

        return np.array(loss_history), metadata

    return training_loop


# Utility functions for loss analysis


def analyze_loss_trajectory(loss_history: np.ndarray) -> dict:
    """
    Analyze loss trajectory to provide insights for hyperparameter tuning.
    """
    if len(loss_history) < 2:
        return {}

    # Compute various statistics
    loss_changes = np.diff(loss_history)

    return {
        "mean_loss": np.mean(loss_history),
        "std_loss": np.std(loss_history),
        "min_loss": np.min(loss_history),
        "max_loss": np.max(loss_history),
        "final_loss": loss_history[-1],
        "total_iterations": len(loss_history),
        "mean_improvement": np.mean(loss_changes),
        "monotonic_decrease": np.all(loss_changes <= 0),
        "oscillating": np.std(loss_changes) > np.abs(np.mean(loss_changes)) * 2,
        "convergence_iteration": _estimate_convergence_point(loss_history),
    }


def _estimate_convergence_point(
    loss_history: np.ndarray, threshold_pct: float = 0.1, window: int = 100
) -> Optional[int]:
    """
    Estimate the iteration where loss converged.

    Pure function using functional style.
    """
    if len(loss_history) < window * 2:
        return None

    # Calculate rolling percentage changes
    changes = np.diff(loss_history)
    denominators = np.maximum(np.abs(loss_history[:-1]), 1e-8)
    pct_changes = np.abs(changes / denominators) * 100

    # Find where moving average stays below threshold
    moving_avg = np.convolve(pct_changes, np.ones(window) / window, mode="valid")

    converged_indices = np.where(moving_avg < threshold_pct)[0]

    if len(converged_indices) > 0:
        return converged_indices[0] + window  # Adjust for convolution offset

    return None


def suggest_hyperparameters(loss_trajectory: dict, model_type: ModelType) -> dict:
    """
    Suggest hyperparameter adjustments based on loss trajectory.

    Pure function that returns recommendations.
    """
    recommendations = {}

    # Check for oscillation
    if loss_trajectory.get("oscillating", False):
        recommendations["learning_rate"] = "decrease"
        recommendations["reason_lr"] = "Loss oscillating, reduce learning rate"

    # Check for slow convergence
    if loss_trajectory.get("convergence_iteration") is None:
        recommendations["n_iter"] = "increase"
        recommendations["reason_n_iter"] = "Model has not converged"

    # Check for premature convergence
    conv_iter = loss_trajectory.get("convergence_iteration", float("inf"))
    if conv_iter < 500 and loss_trajectory.get("final_loss", 0) > 1.0:
        recommendations["learning_rate"] = "increase"
        recommendations["reason_lr"] = "Converged too early, try higher learning rate"

    # Model-specific recommendations
    if model_type in [ModelType.DEEP_GP, ModelType.DEEP_STOCHASTIC]:
        if loss_trajectory.get("total_iterations", 0) < 1500:
            recommendations["n_iter"] = "increase"
            recommendations["reason_n_iter"] = "Deep models need more iterations"

    return recommendations


class MDSPP_Matern:
    def __init__(
        self,
        xin,
        yin,
        nInput,
        nOutput,
        xlb,
        xub,
        num_hidden_dims=3,
        Q=8,
        num_inducing_points=128,
        seed=None,
        gp_lengthscale_bounds=None,
        gp_likelihood_sigma=None,
        linear_mean=True,
        preconditioner_size=100,
        adam_lr=0.1,
        fast_pred_var=False,
        n_iter=2000,
        min_loss_pct_change=1.0,
        batch_size=10,
        return_mean_variance=False,
        use_cuda=False,
        nan="remove",
        top_k=None,
        logger=None,
        **kwargs,
    ):
        if not _has_gpytorch:
            raise RuntimeError(
                "MDSPP_Matern requires the GPyTorch library to be installed."
            )

        self.batch_size = batch_size
        self.linear_mean = linear_mean
        self.fast_pred_var = fast_pred_var
        self.preconditioner_size = preconditioner_size
        self.use_cuda = use_cuda
        self.nInput = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrng = np.where(
            np.isclose(xub - xlb, 0.0, rtol=1e-6, atol=1e-6), 1.0, xub - xlb
        )
        self.logger = logger
        self.return_mean_variance = return_mean_variance

        if nan is not None:
            yin, xin = filter_samples(yin, xin, nan=nan)

        xin, yin = top_k_MO(xin, yin, top_k)

        n_devices = None
        if self.use_cuda:
            n_devices = torch.cuda.device_count()
            if logger is not None:
                logger.info(f"MDSPP_Matern: using {n_devices} GPU devices.")

        N = xin.shape[0]
        xn = np.zeros_like(xin, dtype=np.float32)
        for i in range(N):
            xn[i, :] = (xin[i, :] - self.xlb) / self.xrng
        if nOutput == 1:
            yin = yin.reshape((yin.shape[0], 1))

        self.y_train_mean = np.asarray(
            [np.mean(yin[:, i]) for i in range(yin.shape[1])], dtype=np.float32
        )
        self.y_train_std = np.asarray(
            [
                handle_zeros_in_scale(np.std(yin[:, i], axis=0), copy=False)
                for i in range(yin.shape[1])
            ],
            dtype=np.float32,
        )

        # Remove mean and make unit variance
        yn = np.column_stack(
            tuple(
                (yin[:, i] - self.y_train_mean[i]) / self.y_train_std[i]
                for i in range(yin.shape[1])
            )
        )
        train_x = torch.from_numpy(xn)

        if logger is not None:
            logger.info("MDSPP_Matern: creating regressor for output...")
            for i in range(nOutput):
                logger.info(
                    f"MDSPP_Matern: y_{i + 1} range is {(np.min(yin[:, i]), np.max(yin[:, i]))}"
                )

        train_y = torch.from_numpy(yn.astype(np.float32))

        gp_noise_prior = None
        if gp_likelihood_sigma is not None:
            gp_noise_prior = gpytorch.priors.NormalPrior(
                loc=0.0, scale=gp_likelihood_sigma
            )

        # batch_shape = torch.Size()
        # if batch_size is not None:
        #    batch_shape = torch.Size([batch_size])

        def train(
            nInput,
            nOutput,
            train_x,
            train_y,
            n_iter,
            gp_lengthscale_bounds=None,
            gp_noise_prior=None,
            checkpoint_size=None,
            preconditioner_size=None,
        ):
            from torch.utils.data import TensorDataset, DataLoader

            if self.use_cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            batch_size = self.batch_size
            train_dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=nOutput, noise_prior=gp_noise_prior
            )

            gp_model = GPyTorchMultitaskDSPPMatern(
                train_x=train_x,
                num_tasks=nOutput,
                num_hidden_dims=num_hidden_dims,
                Q=Q,
                num_inducing_points=num_inducing_points,
                likelihood=gp_likelihood,
                lengthscale_bounds=gp_lengthscale_bounds,
                batch_size=batch_size,
                use_cuda=use_cuda,
                n_devices=n_devices,
            )

            if self.use_cuda:
                gp_model = gp_model.cuda()
                gp_likelihood = gp_likelihood.cuda()

            # Find optimal model hyperparameters
            gp_model.train()
            gp_likelihood.train()

            optimizer = torch.optim.Adam(
                gp_model.parameters(), lr=adam_lr
            )  # Includes GaussianLikelihood parameters
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, threshold=0.01
            )

            if logger is not None:
                logger.info("MDSPP_Matern: optimizing regressor...")

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.DeepApproximateMLL(
                gpytorch.mlls.VariationalELBO(
                    gp_model.likelihood, gp_model, num_data=train_y.size(0)
                )
            )

            # Create model-specific configuration
            config = EarlyStoppingConfig.for_model_type(ModelType.DEEP_STOCHASTIC)
            config.threshold_pct = min_loss_pct_change
            early_stopping = AdaptiveEarlyStopping(config)

            batch_loss_log = []

            with ExitStack() as stack:
                if checkpoint_size is not None:
                    stack.enter_context(
                        gpytorch.beta_features.checkpoint_kernel(checkpoint_size)
                    )

                if preconditioner_size is not None:
                    stack.enter_context(
                        gpytorch.settings.max_preconditioner_size(preconditioner_size)
                    )

                for it in range(n_iter):
                    # Zero gradients from previous iteration
                    loss_log = []
                    for x_batch, y_batch in train_loader:
                        with gpytorch.settings.num_likelihood_samples(batch_size):
                            optimizer.zero_grad()
                            if self.use_cuda:
                                x_batch = x_batch.cuda()
                                y_batch = y_batch.cuda()
                            output = gp_model(x_batch)

                            # Calculate loss and backprop gradients
                            loss = -mll(output, y_batch)
                            loss.backward()
                            loss_log.append(loss.item())
                            optimizer.step()
                    mean_loss = np.mean(loss_log)
                    batch_loss_log.append(mean_loss)
                    scheduler.step(mean_loss)
                    if it % 100 == 0:
                        if logger is not None:
                            mean_noise_val = gp_model.likelihood.noise.mean(0)
                            logger.info(
                                f"MDSPP_Matern: iter {it}/{n_iter} - "
                                f"Loss: {mean_loss:.3f}  {mean_noise_val:.3f}"
                            )
                    # Adaptive early stopping check
                    if it >= config.warmup_iterations:
                        should_stop, reason = early_stopping.should_stop(
                            it, np.array(batch_loss_log)
                        )

                        if should_stop:
                            if logger is not None:
                                logger.info(
                                    f"MDSPP_Matern: early stop at iteration {it + 1}: {reason}"
                                )
                                break

            # Analyze training trajectory
            if logger is not None:
                trajectory = analyze_loss_trajectory(np.array(batch_loss_log))
                logger.info("MDSPP_Matern training completed:")
                logger.info(f"  Final loss: {trajectory['final_loss']:.4f}")
                logger.info(f"  Total iterations: {trajectory['total_iterations']}")
                if trajectory.get("convergence_iteration"):
                    logger.info(
                        f"  Converged at: {trajectory['convergence_iteration']}"
                    )

            return gp_model

        if n_devices is not None and n_devices >= 1:
            # Set a large enough preconditioner size to reduce the number of CG iterations run
            self.checkpoint_size = find_best_gpu_setting(
                nInput,
                nOutput,
                train,
                train_x,
                train_y,
                n_devices=n_devices,
                preconditioner_size=self.preconditioner_size,
                logger=logger,
            )
            gp_model = train(
                nInput,
                nOutput,
                train_x,
                train_y,
                n_iter=n_iter,
                gp_lengthscale_bounds=gp_lengthscale_bounds,
                gp_noise_prior=gp_noise_prior,
                checkpoint_size=self.checkpoint_size,
                preconditioner_size=self.preconditioner_size,
            )
        else:
            gp_model = train(
                nInput,
                nOutput,
                train_x,
                train_y,
                n_iter=n_iter,
                gp_lengthscale_bounds=gp_lengthscale_bounds,
                gp_noise_prior=gp_noise_prior,
            )

        del train_x
        del train_y
        if use_cuda:
            torch.cuda.empty_cache()
        self.sm = gp_model

    def predict(self, xin):
        # batch_size = self.batch_size
        # if self.batch_size is None:
        #    batch_size = xin.shape[0]

        x = np.zeros_like(xin, dtype=np.float32)
        if len(x.shape) == 1:
            x = x.reshape((1, self.nInput))
            xin = xin.reshape((1, self.nInput))
        N = x.shape[0]
        y = np.zeros((N, self.nOutput), dtype=np.float32)
        y_var = np.zeros((N, self.nOutput), dtype=np.float32)
        for i in range(N):
            x[i, :] = (xin[i, :] - self.xlb) / self.xrng
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fast_pred_var:
                stack.enter_context(gpytorch.settings.fast_pred_var())
            means, variances = self.sm.predict(x)
            # undo normalization
            if self.use_cuda:
                means = means.cpu()
                variances = variances.cpu()
            y_mean = self.y_train_std * means.numpy() + self.y_train_mean
            y_var[:] = np.multiply(variances, self.y_train_std**2)
            y[:] = y_mean
            del means, variances
        return y, y_var

    def evaluate(self, x):
        mean, var = self.predict(x)
        if self.return_mean_variance:
            return mean, var
        else:
            return mean


class MDGP_Matern:
    def __init__(
        self,
        xin,
        yin,
        nInput,
        nOutput,
        xlb,
        xub,
        num_hidden_dims=3,
        num_inducing_points=128,
        seed=None,
        gp_lengthscale_bounds=None,
        gp_likelihood_sigma=None,
        linear_mean=True,
        preconditioner_size=100,
        adam_lr=0.1,
        fast_pred_var=False,
        n_iter=2000,
        min_loss_pct_change=1.0,
        batch_size=50,
        return_mean_variance=False,
        use_cuda=False,
        nan="remove",
        top_k=None,
        logger=None,
        **kwargs,
    ):
        if not _has_gpytorch:
            raise RuntimeError(
                "MDGP_Matern requires the GPyTorch library to be installed."
            )

        self.batch_size = batch_size
        self.linear_mean = linear_mean
        self.fast_pred_var = fast_pred_var
        self.preconditioner_size = preconditioner_size
        self.use_cuda = use_cuda
        self.nInput = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrng = np.where(
            np.isclose(xub - xlb, 0.0, rtol=1e-6, atol=1e-6), 1.0, xub - xlb
        )
        self.logger = logger
        self.return_mean_variance = return_mean_variance

        if nan is not None:
            yin, xin = filter_samples(yin, xin, nan=nan)

        xin, yin = top_k_MO(xin, yin, top_k)

        n_devices = None
        if self.use_cuda:
            n_devices = torch.cuda.device_count()
            if logger is not None:
                logger.info(f"MDGP_Matern: using {n_devices} GPU devices.")

        N = xin.shape[0]
        xn = np.zeros_like(xin, dtype=np.float32)
        for i in range(N):
            xn[i, :] = (xin[i, :] - self.xlb) / self.xrng
        if nOutput == 1:
            yin = yin.reshape((yin.shape[0], 1))

        self.y_train_mean = np.asarray(
            [np.mean(yin[:, i]) for i in range(yin.shape[1])], dtype=np.float32
        )
        self.y_train_std = np.asarray(
            [
                handle_zeros_in_scale(np.std(yin[:, i], axis=0), copy=False)
                for i in range(yin.shape[1])
            ],
            dtype=np.float32,
        )

        # Remove mean and make unit variance
        yn = np.column_stack(
            tuple(
                (yin[:, i] - self.y_train_mean[i]) / self.y_train_std[i]
                for i in range(yin.shape[1])
            )
        )
        train_x = torch.from_numpy(xn)

        if logger is not None:
            logger.info("MDGP_Matern: creating regressor for output...")
            for i in range(nOutput):
                logger.info(
                    f"MDGP_Matern: y_{i + 1} range is {(np.min(yin[:, i]), np.max(yin[:, i]))}"
                )

        train_y = torch.from_numpy(yn.astype(np.float32))

        gp_noise_prior = None
        if gp_likelihood_sigma is not None:
            gp_noise_prior = gpytorch.priors.NormalPrior(
                loc=0.0, scale=gp_likelihood_sigma
            )

        # batch_shape = torch.Size()
        # if batch_size is not None:
        #    batch_shape = torch.Size([batch_size])

        def train(
            nInput,
            nOutput,
            train_x,
            train_y,
            n_iter,
            gp_lengthscale_bounds=None,
            gp_noise_prior=None,
            checkpoint_size=None,
            preconditioner_size=None,
        ):
            from torch.utils.data import TensorDataset, DataLoader

            if self.use_cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            batch_size = self.batch_size
            train_dataset = TensorDataset(train_x, train_y)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=nOutput, noise_prior=gp_noise_prior
            )

            gp_model = GPyTorchMultitaskDeepGPMatern(
                train_x=train_x,
                num_tasks=nOutput,
                num_hidden_dims=num_hidden_dims,
                num_inducing_points=num_inducing_points,
                likelihood=gp_likelihood,
                lengthscale_bounds=gp_lengthscale_bounds,
                batch_size=batch_size,
                use_cuda=use_cuda,
                n_devices=n_devices,
            )

            if self.use_cuda:
                gp_model = gp_model.cuda()
                gp_likelihood = gp_likelihood.cuda()

            # Find optimal model hyperparameters
            gp_model.train()
            gp_likelihood.train()

            optimizer = torch.optim.Adam(
                gp_model.parameters(), lr=adam_lr
            )  # Includes GaussianLikelihood parameters
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=3, threshold=0.01
            )

            if logger is not None:
                logger.info("MDGP_Matern: optimizing regressor...")

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.DeepApproximateMLL(
                gpytorch.mlls.VariationalELBO(
                    gp_model.likelihood, gp_model, num_data=train_y.size(0)
                )
            )

            # Create model-specific configuration
            config = EarlyStoppingConfig.for_model_type(ModelType.DEEP_GP)
            config.threshold_pct = min_loss_pct_change
            early_stopping = AdaptiveEarlyStopping(config)

            batch_loss_log = []

            with ExitStack() as stack:
                if checkpoint_size is not None:
                    stack.enter_context(
                        gpytorch.beta_features.checkpoint_kernel(checkpoint_size)
                    )

                if preconditioner_size is not None:
                    stack.enter_context(
                        gpytorch.settings.max_preconditioner_size(preconditioner_size)
                    )

                for it in range(n_iter):
                    # Zero gradients from previous iteration
                    loss_log = []
                    for x_batch, y_batch in train_loader:
                        with gpytorch.settings.num_likelihood_samples(batch_size):
                            optimizer.zero_grad()
                            if self.use_cuda:
                                x_batch = x_batch.cuda()
                                y_batch = y_batch.cuda()
                            output = gp_model(x_batch)

                            # Calculate loss and backprop gradients
                            loss = -mll(output, y_batch)
                            loss.backward()
                            loss_log.append(loss.item())
                            optimizer.step()
                    mean_loss = np.mean(loss_log)
                    batch_loss_log.append(mean_loss)
                    scheduler.step(mean_loss)
                    if it % 100 == 0:
                        if logger is not None:
                            mean_noise_val = gp_model.likelihood.noise.mean(0)
                            logger.info(
                                f"MDGP_Matern: iter {it}/{n_iter} - "
                                f"Loss: {mean_loss:.3f}  noise:  {mean_noise_val:.3f}"
                            )

                    if it >= config.warmup_iterations:
                        should_stop, reason = early_stopping.should_stop(
                            it, np.array(batch_loss_log), compute_validation=None
                        )

                        if should_stop:
                            if logger is not None:
                                logger.info(
                                    f"MDGP_Matern: early stop at iteration {it + 1}: {reason}"
                                )
                                break

            # Analyze training trajectory
            if logger is not None:
                trajectory = analyze_loss_trajectory(np.array(batch_loss_log))
                logger.info("MDGP_Matern training completed:")
                logger.info(f"  Final loss: {trajectory['final_loss']:.4f}")
                logger.info(f"  Total iterations: {trajectory['total_iterations']}")
                if trajectory.get("convergence_iteration"):
                    logger.info(
                        f"  Converged at: {trajectory['convergence_iteration']}"
                    )

            return gp_model

        if n_devices is not None and n_devices >= 1:
            # Set a large enough preconditioner size to reduce the number of CG iterations run
            self.checkpoint_size = find_best_gpu_setting(
                nInput,
                nOutput,
                train,
                train_x,
                train_y,
                n_devices=n_devices,
                preconditioner_size=self.preconditioner_size,
                logger=logger,
            )
            gp_model = train(
                nInput,
                nOutput,
                train_x,
                train_y,
                n_iter=n_iter,
                gp_lengthscale_bounds=gp_lengthscale_bounds,
                gp_noise_prior=gp_noise_prior,
                checkpoint_size=self.checkpoint_size,
                preconditioner_size=self.preconditioner_size,
            )
        else:
            gp_model = train(
                nInput,
                nOutput,
                train_x,
                train_y,
                n_iter=n_iter,
                gp_lengthscale_bounds=gp_lengthscale_bounds,
                gp_noise_prior=gp_noise_prior,
            )

        del train_x
        del train_y
        if use_cuda:
            torch.cuda.empty_cache()
        self.sm = gp_model

    def predict(self, xin):
        # batch_size = self.batch_size
        # if self.batch_size is None:
        #    batch_size = xin.shape[0]

        x = np.zeros_like(xin, dtype=np.float32)
        if len(x.shape) == 1:
            x = x.reshape((1, self.nInput))
            xin = xin.reshape((1, self.nInput))
        N = x.shape[0]
        y = np.zeros((N, self.nOutput), dtype=np.float32)
        for i in range(N):
            x[i, :] = (xin[i, :] - self.xlb) / self.xrng
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fast_pred_var:
                stack.enter_context(gpytorch.settings.fast_pred_var())
            means, variances = self.sm.predict(x)
            if self.use_cuda:
                means = means.cpu()
                variances = variances.cpu()
            # undo normalization
            y_mean = self.y_train_std * means.numpy() + self.y_train_mean
            y_var = np.multiply(variances, self.y_train_std**2)
            y[:] = y_mean
            del means, variances
        return y, y_var

    def evaluate(self, x):
        mean, var = self.predict(x)
        if self.return_mean_variance:
            return mean, var
        else:
            return mean


class MEGP_Matern:
    def __init__(
        self,
        xin,
        yin,
        nInput,
        nOutput,
        xlb,
        xub,
        seed=None,
        gp_lengthscale_bounds=None,
        gp_likelihood_sigma=None,
        batch_size=None,
        preconditioner_size=100,
        adam_lr=0.01,
        fast_pred_var=False,
        n_iter=5000,
        min_loss_pct_change=0.1,
        return_mean_variance=False,
        use_cuda=False,
        nan="remove",
        top_k=None,
        logger=None,
        **kwargs,
    ):
        if not _has_gpytorch:
            raise RuntimeError(
                "MEGP_Matern requires the GPyTorch library to be installed."
            )

        self.batch_size = batch_size
        self.fast_pred_var = fast_pred_var
        self.preconditioner_size = preconditioner_size
        self.checkpoint_size = None
        self.use_cuda = use_cuda
        self.nInput = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrng = np.where(
            np.isclose(xub - xlb, 0.0, rtol=1e-6, atol=1e-6), 1.0, xub - xlb
        )
        self.logger = logger
        self.return_mean_variance = return_mean_variance

        if nan is not None:
            yin, xin = filter_samples(yin, xin, nan=nan)

        xin, yin = top_k_MO(xin, yin, top_k)

        n_devices = None
        if self.use_cuda:
            n_devices = torch.cuda.device_count()
            if logger is not None:
                logger.info(f"MEGP_Matern: using {n_devices} GPU devices.")

        N = xin.shape[0]
        xn = np.zeros_like(xin, dtype=np.float32)
        for i in range(N):
            xn[i, :] = (xin[i, :] - self.xlb) / self.xrng
        if nOutput == 1:
            yin = yin.reshape((yin.shape[0], 1))

        self.y_train_mean = np.asarray(
            [np.mean(yin[:, i]) for i in range(yin.shape[1])], dtype=np.float32
        )
        self.y_train_std = np.asarray(
            [
                handle_zeros_in_scale(np.std(yin[:, i], axis=0), copy=False)
                for i in range(yin.shape[1])
            ],
            dtype=np.float32,
        )

        # Remove mean and make unit variance
        yn = np.column_stack(
            tuple(
                (yin[:, i] - self.y_train_mean[i]) / (self.y_train_std[i] + 1e-12)
                for i in range(yin.shape[1])
            )
        )

        train_x = torch.from_numpy(xn)

        if logger is not None:
            logger.info("MEGP_Matern: creating regressor for output...")
            for i in range(nOutput):
                logger.info(
                    f"MEGP_Matern: y_{i + 1} range is {(np.min(yin[:, i]), np.max(yin[:, i]))}"
                )

        train_y = torch.from_numpy(yn.astype(np.float32))

        gp_noise_prior = None
        if gp_likelihood_sigma is not None:
            gp_noise_prior = gpytorch.priors.NormalPrior(
                loc=0.0, scale=gp_likelihood_sigma
            )

        def train(
            nInput,
            nOutput,
            train_x,
            train_y,
            n_iter,
            gp_lengthscale_bounds=None,
            gp_noise_prior=None,
            checkpoint_size=None,
            preconditioner_size=None,
        ):
            if self.use_cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=nOutput, noise_prior=gp_noise_prior
            )

            gp_model = GPyTorchMultitaskExactGPModelMatern(
                train_x=train_x,
                train_y=train_y,
                num_tasks=nOutput,
                ard_num_dims=nInput,
                likelihood=gp_likelihood,
                lengthscale_bounds=gp_lengthscale_bounds,
                n_devices=n_devices,
            )

            if self.use_cuda:
                gp_model = gp_model.cuda()
                gp_likelihood = gp_likelihood.cuda()

            # Find optimal model hyperparameters
            gp_model.train()
            gp_likelihood.train()

            optimizer = torch.optim.Adam(
                gp_model.parameters(), lr=adam_lr
            )  # Includes GaussianLikelihood parameters

            if logger is not None:
                logger.info("MEGP_Matern: optimizing regressor...")

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_likelihood, gp_model)

            # Create adaptive early stopping
            config = EarlyStoppingConfig.for_model_type(ModelType.EXACT_GP)
            config.threshold_pct = min_loss_pct_change
            early_stopping = AdaptiveEarlyStopping(config)

            loss_log = []

            with ExitStack() as stack:
                if checkpoint_size is not None:
                    stack.enter_context(
                        gpytorch.beta_features.checkpoint_kernel(checkpoint_size)
                    )
                if preconditioner_size is not None:
                    stack.enter_context(
                        gpytorch.settings.max_preconditioner_size(preconditioner_size)
                    )

                for it in range(n_iter):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = gp_model(train_x)
                    # Calculate loss and backprop gradients
                    loss = -mll(output, train_y)
                    loss.backward()
                    optimizer.step()
                    loss_value = loss.item()
                    loss_log.append(loss_value)
                    if it % 100 == 0:
                        if logger is not None:
                            noise_val = gp_model.likelihood.noise.item()
                            logger.info(
                                f"MEGP_Matern: iter {it}/{n_iter} - "
                                f"Loss: {loss_value:.3f}  noise: {noise_val:.3f}"
                            )
                    if it >= config.warmup_iterations:
                        should_stop, reason = early_stopping.should_stop(
                            it,
                            np.array(loss_log),
                            compute_validation=None,  # Add validation if available
                        )

                        if should_stop:
                            if logger is not None:
                                logger.info(
                                    f"MEGP_Matern: early stop at iteration {it + 1}: {reason}"
                                )
                            break

            # Analyze training trajectory
            if logger is not None:
                trajectory = analyze_loss_trajectory(np.array(loss_log))
                logger.info("MEGP_Matern training completed:")
                logger.info(f"  Final loss: {trajectory['final_loss']:.4f}")
                logger.info(f"  Total iterations: {trajectory['total_iterations']}")
                if trajectory.get("convergence_iteration"):
                    logger.info(
                        f"  Converged at: {trajectory['convergence_iteration']}"
                    )

            return gp_model

        if n_devices is not None and n_devices >= 1:
            # Set a large enough preconditioner size to reduce the number of CG iterations run
            self.checkpoint_size = find_best_gpu_setting(
                nInput,
                nOutput,
                train,
                train_x,
                train_y,
                n_devices=n_devices,
                preconditioner_size=self.preconditioner_size,
                logger=logger,
            )
            gp_model = train(
                nInput,
                nOutput,
                train_x,
                train_y,
                n_iter=n_iter,
                gp_lengthscale_bounds=gp_lengthscale_bounds,
                gp_noise_prior=gp_noise_prior,
                checkpoint_size=self.checkpoint_size,
                preconditioner_size=self.preconditioner_size,
            )

        else:
            gp_model = train(
                nInput,
                nOutput,
                train_x,
                train_y,
                n_iter=n_iter,
                gp_lengthscale_bounds=gp_lengthscale_bounds,
                gp_noise_prior=gp_noise_prior,
            )

        del train_x
        del train_y
        if use_cuda:
            torch.cuda.empty_cache()
        self.sm = gp_model

    def predict(self, xin):
        from torch.utils.data import DataLoader

        batch_size = self.batch_size
        if self.batch_size is None:
            batch_size = xin.shape[0]

        x = np.zeros_like(xin, dtype=np.float32)
        if len(x.shape) == 1:
            x = x.reshape((1, self.nInput))
            xin = xin.reshape((1, self.nInput))
        N = x.shape[0]
        y = np.zeros((N, self.nOutput), dtype=np.float32)
        for i in range(N):
            x[i, :] = (xin[i, :] - self.xlb) / self.xrng
        x = torch.from_numpy(x)
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fast_pred_var:
                stack.enter_context(gpytorch.settings.fast_pred_var())
            if self.checkpoint_size is not None:
                stack.enter_context(
                    gpytorch.beta_features.checkpoint_kernel(self.checkpoint_size)
                )
            self.sm.eval()
            self.sm.likelihood.eval()

            means = []
            variances = []
            in_loader = DataLoader(x, batch_size=batch_size, shuffle=False)
            for x_batch in in_loader:
                if self.use_cuda:
                    x_batch = x_batch.cuda()
                f_preds = self.sm.likelihood(self.sm(x_batch))
                mean, var = f_preds.mean, f_preds.variance
                means.append(mean)
                variances.append(var)
            means = torch.cat(means)
            variances = torch.cat(variances)
            # undo normalization
            if self.use_cuda:
                means = means.cpu()
                variances = variances.cpu()
            y_mean = self.y_train_std * means.numpy() + self.y_train_mean
            y_var = np.multiply(variances.numpy(), self.y_train_std**2)
            y[:] = y_mean
            del means, variances
        return y, y_var

    def evaluate(self, x):
        mean, var = self.predict(x)
        if self.return_mean_variance:
            return mean, var
        else:
            return mean


class EGP_Matern:
    def __init__(
        self,
        xin,
        yin,
        nInput,
        nOutput,
        xlb,
        xub,
        seed=None,
        gp_lengthscale_bounds=None,
        gp_likelihood_sigma=None,
        preconditioner_size=100,
        adam_lr=0.01,
        fast_pred_var=True,
        n_iter=5000,
        min_loss_pct_change=0.1,
        return_mean_variance=False,
        batch_size=None,
        use_cuda=False,
        nan="remove",
        top_k=None,
        logger=None,
        **kwargs,
    ):
        if not _has_gpytorch:
            raise RuntimeError(
                "EGP_Matern requires the GPyTorch library to be installed."
            )

        self.fast_pred_var = fast_pred_var
        self.preconditioner_size = preconditioner_size
        self.checkpoint_size = None
        self.use_cuda = use_cuda
        self.nInput = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrng = np.where(
            np.isclose(xub - xlb, 0.0, rtol=1e-6, atol=1e-6), 1.0, xub - xlb
        )

        self.logger = logger
        self.return_mean_variance = return_mean_variance

        if nan is not None:
            yin, xin = filter_samples(yin, xin, nan=nan)

        xin, yin = top_k_MO(xin, yin, top_k)

        n_devices = None
        if self.use_cuda:
            n_devices = torch.cuda.device_count()
            if logger is not None:
                logger.info(f"EGP_Matern: using {n_devices} GPU devices.")

        N = xin.shape[0]
        xn = np.zeros_like(xin, dtype=np.float32)
        for i in range(N):
            xn[i, :] = (xin[i, :] - self.xlb) / self.xrng
        if nOutput == 1:
            yin = yin.reshape((yin.shape[0], 1))

        self.y_train_mean = np.asarray(
            [np.mean(yin[:, i]) for i in range(yin.shape[1])], dtype=np.float32
        )
        self.y_train_std = np.asarray(
            [
                handle_zeros_in_scale(np.std(yin[:, i], axis=0), copy=False)
                for i in range(yin.shape[1])
            ],
            dtype=np.float32,
        )

        # Remove mean and make unit variance
        yn = np.column_stack(
            tuple(
                (yin[:, i] - self.y_train_mean[i]) / self.y_train_std[i]
                for i in range(yin.shape[1])
            )
        )
        train_x = torch.from_numpy(xn)

        gp_noise_prior = None

        if gp_likelihood_sigma is not None:
            gp_noise_prior = gpytorch.priors.NormalPrior(
                loc=0.0, scale=gp_likelihood_sigma
            )

        batch_shape = torch.Size()
        if batch_size is not None:
            batch_shape = torch.Size([batch_size])

        def train(
            nInput,
            nOutput,
            train_x,
            train_y,
            n_iter,
            gp_lengthscale_bounds=None,
            gp_noise_prior=None,
            checkpoint_size=None,
            preconditioner_size=None,
        ):
            gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_prior=gp_noise_prior, batch_shape=batch_shape
            )

            gp_model = GPyTorchExactGPModelMatern(
                train_x=train_x,
                train_y=train_y,
                ard_num_dims=nInput,
                likelihood=gp_likelihood,
                lengthscale_bounds=gp_lengthscale_bounds,
                batch_size=batch_size,
            )

            if self.use_cuda:
                train_x = train_x.cuda()
                train_y = train_y.cuda()
                gp_model = gp_model.cuda()
                gp_likelihood = gp_likelihood.cuda()

            # Find optimal model hyperparameters
            gp_model.train()
            gp_likelihood.train()

            optimizer = torch.optim.Adam(
                gp_model.parameters(), lr=adam_lr
            )  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp_likelihood, gp_model)

            # Create adaptive early stopping
            config = EarlyStoppingConfig.for_model_type(ModelType.EXACT_GP)
            config.threshold_pct = min_loss_pct_change
            early_stopping = AdaptiveEarlyStopping(config)

            loss_log = []

            with ExitStack() as stack:
                if checkpoint_size is not None:
                    stack.enter_context(
                        gpytorch.beta_features.checkpoint_kernel(checkpoint_size)
                    )
                if preconditioner_size is not None:
                    stack.enter_context(
                        gpytorch.settings.max_preconditioner_size(preconditioner_size)
                    )
                for it in range(n_iter):
                    # Zero gradients from previous iteration
                    optimizer.zero_grad()
                    # Output from model
                    output = gp_model(train_x)
                    # Calculate loss and backprop gradients
                    if batch_size is None:
                        loss = -mll(output, train_y)
                    else:
                        loss = -mll(output, train_y).sum()
                    loss.backward()
                    loss_value = loss.item()
                    loss_log.append(loss_value)
                    optimizer.step()
                    if it % 100 == 0:
                        if logger is not None:
                            noise_val = gp_model.likelihood.noise.item()
                            logger.info(
                                f"EGP_Matern: iter {it}/{n_iter} - "
                                f"Loss: {loss_value:.3f}  noise: {noise_val:.3f}"
                            )
                    if it >= config.warmup_iterations:
                        should_stop, reason = early_stopping.should_stop(
                            it,
                            np.array(loss_log),
                            compute_validation=None,  # Add validation if available
                        )

                        if should_stop:
                            if logger is not None:
                                logger.info(
                                    f"EGP_Matern: early stop at iteration {it + 1}: {reason}"
                                )
                            break

            # Analyze training trajectory
            if logger is not None:
                trajectory = analyze_loss_trajectory(np.array(loss_log))
                logger.info("EGP_Matern training completed:")
                logger.info(f"  Final loss: {trajectory['final_loss']:.4f}")
                logger.info(f"  Total iterations: {trajectory['total_iterations']}")
                if trajectory.get("convergence_iteration"):
                    logger.info(
                        f"  Converged at: {trajectory['convergence_iteration']}"
                    )

            return gp_model

        smlist = []
        for i in range(nOutput):
            if logger is not None:
                logger.info(
                    f"EGP_Matern: creating regressor for output {i + 1} of {nOutput}..."
                )
                logger.info(
                    f"EGP_Matern: y_{i} range is {(np.min(yin[:, i]), np.max(yin[:, i]))}..."
                )

            train_y = torch.from_numpy(yn[:, i].reshape((-1,)).astype(np.float32))

            if logger is not None:
                logger.info(
                    f"EGP_Matern: optimizing regressor for output {i + 1} of {nOutput}..."
                )

            if n_devices is not None and n_devices >= 1:
                # Set a large enough preconditioner size to reduce the number of CG iterations run
                self.checkpoint_size = find_best_gpu_setting(
                    nInput,
                    1,
                    train,
                    train_x,
                    train_y,
                    n_devices=n_devices,
                    preconditioner_size=self.preconditioner_size,
                    logger=logger,
                )
                gp_model = train(
                    nInput,
                    1,
                    train_x,
                    train_y,
                    n_iter=n_iter,
                    gp_lengthscale_bounds=gp_lengthscale_bounds,
                    gp_noise_prior=gp_noise_prior,
                    checkpoint_size=self.checkpoint_size,
                    preconditioner_size=self.preconditioner_size,
                )
            else:
                gp_model = train(
                    nInput,
                    1,
                    train_x,
                    train_y,
                    n_iter=n_iter,
                    gp_lengthscale_bounds=gp_lengthscale_bounds,
                    gp_noise_prior=gp_noise_prior,
                )

            del train_y
            smlist.append(gp_model)

        del train_x

        if use_cuda:
            torch.cuda.empty_cache()
        self.smlist = smlist

    def predict(self, xin):
        x = np.zeros_like(xin, dtype=np.float32)
        if len(x.shape) == 1:
            x = x.reshape((1, self.nInput))
            xin = xin.reshape((1, self.nInput))
        N = x.shape[0]
        y = np.zeros((N, self.nOutput), dtype=np.float32)
        y_vars = np.zeros((N, self.nOutput), dtype=np.float32)
        for i in range(N):
            x[i, :] = (xin[i, :] - self.xlb) / self.xrng
        x = torch.from_numpy(x)
        if self.use_cuda:
            x = x.cuda()

        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if self.fast_pred_var:
                stack.enter_context(gpytorch.settings.fast_pred_var())
            if self.checkpoint_size is not None:
                stack.enter_context(
                    gpytorch.beta_features.checkpoint_kernel(self.checkpoint_size)
                )
            for i in range(self.nOutput):
                self.smlist[i].eval()
                self.smlist[i].likelihood.eval()
                f_preds = self.smlist[i].likelihood(self.smlist[i](x))
                mean, var = f_preds.mean, f_preds.variance
                # undo normalization
                if self.use_cuda:
                    mean = mean.cpu()
                    var = var.cpu()
                y_mean = (
                    self.y_train_std[i] * np.reshape(mean.numpy(), [-1])
                    + self.y_train_mean[i]
                )
                y_var = np.multiply(var, self.y_train_std[i] ** 2)
                y[:, i] = y_mean
                y_vars[:, i] = y_var
                del mean
                del var
        return y, y_vars

    def evaluate(self, x):
        mean, var = self.predict(x)
        if self.return_mean_variance:
            return mean, var
        else:
            return mean
