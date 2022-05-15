import timeit
import copy
import numpy as np
from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel

try:
    import gpflow
    import tensorflow as tf
    import tensorflow_probability as tfp
    from gpflow.utilities import print_summary
    from gpflow.ci_utils import ci_niter, ci_range
    from gpflow.models import VGP, GPR, SVGP
    from gpflow.optimizers import NaturalGradient
    from gpflow.optimizers.natgrad import XiSqrtMeanVar
    gpflow.config.set_default_float(np.float64)
    gpflow.config.set_default_jitter(10e-3)
    tf.keras.backend.set_floatx('float64')

    def bounded_parameter(low, high, value, **kwargs):
        """Returns Parameter with optimization bounds."""
        sigmoid = tfp.bijectors.Sigmoid(low, high)
        parameter = gpflow.Parameter(value, transform=sigmoid, dtype=tf.float64, **kwargs)
        return parameter


except:
    _has_gpflow = False
else:
    _has_gpflow = True

    

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

    
class SIV_Matern:

    def __init__(self, xin, yin, nInput, nOutput, xlb, xub, seed=None, batch_size=100, inducing_fraction=0.2, min_inducing=100, gp_lengthscale_bounds=(1e-6, 100.0), gp_likelihood_sigma=1.0e-4, natgrad_gamma=0.1, adam_lr=0.01, n_iter=30000, min_elbo_pct_change=1.0, num_latent_gps=None, logger=None):
        if not _has_gpflow:
            raise RuntimeError('SIV_Matern requires the GPflow library to be installed.')
            
        self.nInput  = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrng = np.where(np.isclose(xub - xlb, 0., rtol=1e-6, atol=1e-6), 1., xub - xlb) 

        self.logger = logger

        N = xin.shape[0]
        D = xin.shape[1]
        xn = np.zeros_like(xin)
        for i in range(N):
            xn[i,:] = (xin[i,:] - self.xlb) / self.xrng
        if nOutput == 1:
            yin = yin.reshape((yin.shape[0],1))
        if num_latent_gps is None:
            num_latent_gps = nOutput
            
        self.y_train_mean = np.asarray([np.mean(yin[:,i]) for i in range(yin.shape[1])], dtype=np.float32)
        self.y_train_std = np.asarray([handle_zeros_in_scale(np.std(yin[:,i], axis=0), copy=False)
                                       for i in range(yin.shape[1])], dtype=np.float32)

        # Remove mean and make unit variance
        yn = np.column_stack(tuple((yin[:,i] - self.y_train_mean[i]) / self.y_train_std[i] for i in range(yin.shape[1])))
            
        adam_opt=tf.optimizers.Adam(adam_lr)
        natgrad_opt=NaturalGradient(gamma=natgrad_gamma)
        autotune = tf.data.experimental.AUTOTUNE
        
        if logger is not None:
            logger.info(f"SIV_Matern: creating regressor for output...")
            for i in range(nOutput):
                logger.info(f"SIV_Matern: y_{i+1} range is {(np.min(yin[:,i]), np.max(yin[:,i]))}")

        data=(np.asarray(xn, dtype=np.float64), yn.astype(np.float64))

        M = int(round(inducing_fraction * N))

        if M < min_inducing:
            Z = xn.copy()
        else:
            Z = xn[np.random.choice(N, size=M, replace=False), :].copy()  # Initialize inducing locations to M random inputs
        iv = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(Z)
        )
        kernel = gpflow.kernels.Matern52()
        gp_kernel = gpflow.kernels.SharedIndependent(kernel, output_dim=nOutput)
        gp_likelihood=gpflow.likelihoods.Gaussian(variance=gp_likelihood_sigma)
        gp_model = gpflow.models.SVGP(
            inducing_variable=iv,
            kernel=gp_kernel,
            likelihood=gp_likelihood,
            num_data=N,
            num_latent_gps=num_latent_gps)

        gp_model.kernel.kernel.lengthscales = bounded_parameter(np.asarray([gp_lengthscale_bounds[0]]*nInput, dtype=np.float64),
                                                                np.asarray([gp_lengthscale_bounds[1]]*nInput, dtype=np.float64),
                                                                np.ones(nInput, dtype=np.float64), trainable=True,
                                                                name='lengthscales')

        gpflow.set_trainable(gp_model.q_mu, False)
        gpflow.set_trainable(gp_model.q_sqrt, False)
        gpflow.set_trainable(gp_model.inducing_variable, False)

        if logger is not None:
            logger.info(f"SIV_Matern: optimizing regressor...")

        variational_params = [(gp_model.q_mu, gp_model.q_sqrt)]

        data_minibatch = (
            tf.data.Dataset.from_tensor_slices(data)
            .prefetch(autotune)
            .repeat()
            .shuffle(N)
            .batch(batch_size)
        )
        data_minibatch_it = iter(data_minibatch)
        svgp_natgrad_loss = gp_model.training_loss_closure(data_minibatch_it, compile=True)
            
        @tf.function
        def optim_step():
            natgrad_opt.minimize(svgp_natgrad_loss, var_list=variational_params)
            adam_opt.minimize(svgp_natgrad_loss, var_list=gp_model.trainable_variables)
                
        iterations = ci_niter(n_iter)
        elbo_log = []
        diff_kernel = np.array([1,-1])
        for it in range(iterations):
            optim_step()
            if (it % 10 == 0):
                likelihood = -svgp_natgrad_loss().numpy()
                elbo_log.append(likelihood)
            if (it % 1000 == 0):
                logger.info(f"SIV_Matern: iteration {it} likelihood: {likelihood:.04f}")
            if it >= 2000:
                elbo_change = np.convolve(elbo_log, diff_kernel, 'same')[1:]
                elbo_pct_change = (elbo_change / np.abs(elbo_log[1:]))*100
                mean_elbo_pct_change = np.mean(elbo_pct_change[-100:])
                if (it % 1000 == 0):
                    logger.info(f"SIV_Matern: iteration {it} mean elbo pct change: {mean_elbo_pct_change:.04f}")
                if mean_elbo_pct_change < min_elbo_pct_change:
                    logger.info(f"SIV_Matern: likelihood change at iteration {it+1} is less than {min_elbo_pct_change} percent")
                    break
        print_summary(gp_model)
        self.sm = gp_model

    def predict(self,xin):
        x = np.zeros_like(xin, dtype=np.float64)
        if len(x.shape) == 1:
            x = x.reshape((1,self.nInput))
            xin = xin.reshape((1,self.nInput))
        N = x.shape[0]
        y = np.zeros((N,self.nOutput), dtype=np.float32)
        for i in range(N):
            x[i,:] = (xin[i,:] - self.xlb) / self.xrng

        mean, var = self.sm.predict_y(x)
        # undo normalization
        y_mean = self.y_train_std * mean + self.y_train_mean
        y = np.zeros((N, self.nOutput), dtype=np.float32)
        y[:] = y_mean
        
        return y

    def evaluate(self,x):
        return self.predict(x)

    
class SVGP_Matern:
    def __init__(self, xin, yin, nInput, nOutput, xlb, xub, seed=None, batch_size=100, inducing_fraction=0.2, min_inducing=100, gp_lengthscale_bounds=(1e-6, 100.0), gp_likelihood_sigma=1.0e-4, natgrad_gamma=0.1, adam_lr=0.01, n_iter=30000, min_elbo_pct_change=1.0, logger=None):
        if not _has_gpflow:
            raise RuntimeError('SVGP_Matern requires the GPflow library to be installed.')
            
        self.nInput  = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrng = np.where(np.isclose(xub - xlb, 0., rtol=1e-6, atol=1e-6), 1., xub - xlb) 

        self.logger = logger

        N = xin.shape[0]
        D = xin.shape[1]
        xn = np.zeros_like(xin)
        for i in range(N):
            xn[i,:] = (xin[i,:] - self.xlb) / self.xrng
        if nOutput == 1:
            yin = yin.reshape((yin.shape[0],1))

        self.y_train_mean = np.asarray([np.mean(yin[:,i]) for i in range(yin.shape[1])], dtype=np.float32)
        self.y_train_std = np.asarray([handle_zeros_in_scale(np.std(yin[:,i], axis=0), copy=False)
                                       for i in range(yin.shape[1])], dtype=np.float32)

        # Remove mean and make unit variance
        yn = np.column_stack(tuple((yin[:,i] - self.y_train_mean[i]) / self.y_train_std[i] for i in range(yin.shape[1])))
            
        adam_opt=tf.optimizers.Adam(adam_lr)
        natgrad_opt=NaturalGradient(gamma=natgrad_gamma)
        autotune = tf.data.experimental.AUTOTUNE
        
        smlist = []
        for i in range(nOutput):
            if logger is not None:
                logger.info(f"SVGP_Matern: creating regressor for output {i+1} of {nOutput}...")
                logger.info(f"SVGP_Matern: y_{i} range is {(np.min(yin[:,i]), np.max( yin[:,i]))}...")

            data=(np.asarray(xn, dtype=np.float64), yn[:, i].reshape((-1,1)).astype(np.float64))

            M = int(round(inducing_fraction * N))
            #Z = tf.random.uniform((M, D))  # Initialize inducing locations to M random inputs
            if M < min_inducing:
                Z = xn.copy()
            else:
                Z = xn[np.random.choice(N, size=M, replace=False), :].copy()  # Initialize inducing locations to M random inputs
            gp_kernel=gpflow.kernels.Matern52()
            gp_likelihood=gpflow.likelihoods.Gaussian(variance=gp_likelihood_sigma)
            gp_model = gpflow.models.SVGP(
                inducing_variable=Z,
                kernel=gp_kernel,
                likelihood=gp_likelihood,
                num_data=N
            )
            gp_model.kernel.lengthscales = bounded_parameter(np.asarray([gp_lengthscale_bounds[0]]*nInput, dtype=np.float64),
                                                             np.asarray([gp_lengthscale_bounds[1]]*nInput, dtype=np.float64),
                                                             np.ones(nInput, dtype=np.float64), trainable=True,
                                                             name='lengthscales')

            gpflow.set_trainable(gp_model.q_mu, False)
            gpflow.set_trainable(gp_model.q_sqrt, False)
            gpflow.set_trainable(gp_model.inducing_variable, False)
            
            if logger is not None:
                logger.info(f"SVGP_Matern: optimizing regressor for output {i+1} of {nOutput}...")

            variational_params = [(gp_model.q_mu, gp_model.q_sqrt)]

            data_minibatch = (
                tf.data.Dataset.from_tensor_slices(data)
                .prefetch(autotune)
                .repeat()
                .shuffle(N)
                .batch(batch_size)
            )
            data_minibatch_it = iter(data_minibatch)
            svgp_natgrad_loss = gp_model.training_loss_closure(data_minibatch_it, compile=True)
            
            @tf.function
            def optim_step():
                adam_opt.minimize(svgp_natgrad_loss, var_list=gp_model.trainable_variables)
                natgrad_opt.minimize(svgp_natgrad_loss, var_list=variational_params)
                
            iterations = ci_niter(n_iter)
            elbo_log = []
            diff_kernel = np.array([1,-1])
            for it in range(iterations):
                optim_step()
                if (it % 10 == 0):
                    likelihood = -svgp_natgrad_loss().numpy()
                    elbo_log.append(likelihood)
                if (it % 1000 == 0):
                    logger.info(f"SVGP_Matern: iteration {it} likelihood: {likelihood:.04f}")
                if it >= 2000:
                    elbo_change = np.convolve(elbo_log, diff_kernel, 'same')[1:]
                    elbo_pct_change = (elbo_change / np.abs(elbo_log[1:]))*100
                    mean_elbo_pct_change = np.mean(elbo_pct_change[-100:])
                    if (it % 1000 == 0):
                        logger.info(f"SVGP_Matern: iteration {it} mean elbo pct change: {mean_elbo_pct_change:.04f}")
                    if mean_elbo_pct_change < min_elbo_pct_change:
                        logger.info(f"SVGP_Matern: likelihood change at iteration {it+1} is less than {min_elbo_pct_change} percent")
                        break
            print_summary(gp_model)
            #assert(opt_log.success)
            smlist.append(gp_model)
        self.smlist = smlist

    def predict(self,xin):
        x = np.zeros_like(xin, dtype=np.float64)
        if len(x.shape) == 1:
            x = x.reshape((1,self.nInput))
            xin = xin.reshape((1,self.nInput))
        N = x.shape[0]
        y = np.zeros((N,self.nOutput), dtype=np.float32)
        for i in range(N):
            x[i,:] = (xin[i,:] - self.xlb) / self.xrng
        for i in range(self.nOutput):
            mean, var = self.smlist[i].predict_y(x)
            # undo normalization
            y_mean = self.y_train_std[i] * tf.reshape(mean, [-1]) + self.y_train_mean[i]
            y[:,i] = tf.cast(y_mean, tf.float32)
        return y

    def evaluate(self,x):
        return self.predict(x)

    
class VGP_Matern:
    def __init__(self, xin, yin, nInput, nOutput, xlb, xub, seed=None, gp_lengthscale_bounds=(1e-6, 100.0), gp_likelihood_sigma=1.0e-4, natgrad_gamma=1.0, adam_lr=0.01, n_iter=3000, min_elbo_pct_change=0.1, logger=None):
        if not _has_gpflow:
            raise RuntimeError('VGP_Matern requires the GPflow library to be installed.')
            
        self.nInput  = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrng = np.where(np.isclose(xub - xlb, 0., rtol=1e-6, atol=1e-6), 1., xub - xlb) 

        self.logger = logger

        N = xin.shape[0]
        xn = np.zeros_like(xin)
        for i in range(N):
            xn[i,:] = (xin[i,:] - self.xlb) / self.xrng
        if nOutput == 1:
            yin = yin.reshape((yin.shape[0],1))

        self.y_train_mean = np.asarray([np.mean(yin[:,i]) for i in range(yin.shape[1])], dtype=np.float32)
        self.y_train_std = np.asarray([handle_zeros_in_scale(np.std(yin[:,i], axis=0), copy=False)
                                       for i in range(yin.shape[1])], dtype=np.float32)

        # Remove mean and make unit variance
        yn = np.column_stack(tuple((yin[:,i] - self.y_train_mean[i]) / self.y_train_std[i] for i in range(yin.shape[1])))
            
        adam_opt=tf.optimizers.Adam(adam_lr)
        natgrad_opt=NaturalGradient(gamma=natgrad_gamma)

        smlist = []
        for i in range(nOutput):
            if logger is not None:
                logger.info(f"VGP_Matern: creating regressor for output {i+1} of {nOutput}...")
                logger.info(f"VGP_Matern: y_{i} range is {(np.min(yin[:,i]), np.max(yin[:,i]))}...")
                
            gp_kernel=gpflow.kernels.Matern52()
            gp_likelihood=gpflow.likelihoods.Gaussian(variance=gp_likelihood_sigma)
            gp_model = gpflow.models.VGP(
                data=(np.asarray(xn, dtype=np.float64), yn[:, i].reshape((-1,1)).astype(np.float64)),
                kernel=gp_kernel,
                likelihood=gp_likelihood,

            )
            gp_model.kernel.lengthscales = bounded_parameter(np.asarray([gp_lengthscale_bounds[0]]*nInput, dtype=np.float64),
                                                             np.asarray([gp_lengthscale_bounds[1]]*nInput, dtype=np.float64),
                                                             np.ones(nInput, dtype=np.float64), trainable=True,
                                                             name='lengthscales')
            
            gpflow.set_trainable(gp_model.q_mu, False)
            gpflow.set_trainable(gp_model.q_sqrt, False)

            if logger is not None:
                logger.info(f"VGP_Matern: optimizing regressor for output {i+1} of {nOutput}...")

            variational_params = [(gp_model.q_mu, gp_model.q_sqrt)]
            iterations = ci_niter(n_iter)
            elbo_log = []
            diff_kernel = np.array([1,-1])


            @tf.function
            def optim_step():
                natgrad_opt.minimize(gp_model.training_loss, var_list=variational_params)
                adam_opt.minimize(gp_model.training_loss, var_list=gp_model.trainable_variables)

            for it in range(iterations):
                optim_step()
                likelihood = gp_model.elbo()
                if (it % 100 == 0):
                    logger.info(f"VGP_Matern: iteration {it} likelihood: {likelihood:.04f}")
                elbo_log.append(likelihood)
                if it >= 200:
                    elbo_change = np.convolve(elbo_log, diff_kernel, 'same')[1:]
                    elbo_pct_change = (elbo_change / np.abs(elbo_log[1:]))*100
                    mean_elbo_pct_change = np.mean(elbo_pct_change[-100:])
                    if mean_elbo_pct_change < min_elbo_pct_change:
                        logger.info(f"VGP_Matern: likelihood change at iteration {it+1} is less than {min_elbo_pct_change} percent")
                        break
            print_summary(gp_model)
            #assert(opt_log.success)
            smlist.append(gp_model)
        self.smlist = smlist

    def predict(self,xin):
        x = np.zeros_like(xin, dtype=np.float64)
        if len(x.shape) == 1:
            x = x.reshape((1,self.nInput))
            xin = xin.reshape((1,self.nInput))
        N = x.shape[0]
        y = np.zeros((N,self.nOutput), dtype=np.float32)
        for i in range(N):
            x[i,:] = (xin[i,:] - self.xlb) / self.xrng
        for i in range(self.nOutput):
            mean, var = self.smlist[i].predict_y(x)
            # undo normalization
            y_mean = self.y_train_std[i] * tf.reshape(mean, [-1]) + self.y_train_mean[i]
            y[:,i] = tf.cast(y_mean, tf.float32)
        return y

    def evaluate(self,x):
        return self.predict(x)
    
    

class GPR_Matern:
    def __init__(self, xin, yin, nInput, nOutput, xlb, xub, optimizer="sceua", seed=None, length_scale_bounds=(1e-2, 100.0), anisotropic=False, logger=None):
        self.nInput  = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrg = xub - xlb
        self.logger = logger

        N = xin.shape[0]
        x = np.zeros_like(xin)
        y = np.copy(yin)
        for i in range(N):
            x[i,:] = (xin[i,:] - self.xlb) / self.xrg
        if nOutput == 1:
            y = y.reshape((y.shape[0],1))

        length_scale=0.5
        if anisotropic:
            length_scale=np.asarray([0.5]*nInput)
        kernel = ConstantKernel(1, (0.01, 100)) * Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=2.5) + \
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-4))
        smlist = []
        for i in range(nOutput):
            if logger is not None:
                logger.info(f"GPR_Matern: creating regressor for output {i+1} of {nOutput}...")
                logger.info(f"GPR_Matern: y_{i} range is {(np.min(y[:,i]), np.max(y[:,i]))}...")
            if optimizer == "sceua":
                optf=partial(sceua_optimizer, seed, logger)
            elif optimizer == "dlib":
                optf=partial(dlib_optimizer, logger)
            else:
                optf=partial(sceua_optimizer, seed, logger)                
            #smlist.append(GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=5))
            smlist.append(GaussianProcessRegressor(kernel=kernel, optimizer=optf, normalize_y=True))
            smlist[i].fit(x,y[:,i])
        self.smlist = smlist

    def predict(self,xin):
        x = np.zeros_like(xin)
        if len(x.shape) == 1:
            x = x.reshape((1,self.nInput))
            xin = xin.reshape((1,self.nInput))
        N = x.shape[0]
        y = np.zeros((N,self.nOutput))
        for i in range(N):
            x[i,:] = (xin[i,:] - self.xlb) / self.xrg
        for i in range(self.nOutput):
            y[:,i] = self.smlist[i].predict(x)
        return y

    def evaluate(self,x):
        return self.predict(x)

    
class GPR_RBF:
    def __init__(self, xin, yin, nInput, nOutput, xlb, xub, optimizer="sceua", seed=None, length_scale_bounds=(1e-2, 100.0), anisotropic=False, logger=None):
        self.nInput  = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrg = xub - xlb
        self.logger = logger

        N = x.shape[0]
        x = np.zeros_like(xin)
        y = np.copy(yin)
        for i in range(N):
            x[i,:] = (xin[i,:] - self.xlb) / self.xrg
        if nOutput == 1:
            y = y.reshape((y.shape[0],1))

        length_scale=0.5
        if anisotropic:
            length_scale=np.asarray([0.5]*nInput)
        kernel = ConstantKernel(1, (0.01, 100)) * RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + \
            WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-4))
        smlist = []
        for i in range(nOutput):
            if logger is not None:
                logger.info(f"GPR_RBF: creating regressor for output {i+1} of {nOutput}...")
                logger.info(f"GPR_RBF: y_{i} range is {(np.min(y[:,i]), np.max(y[:,i]))}...")
            if optimizer == "sceua":
                optf=partial(sceua_optimizer, seed, logger)
            elif optimizer == "dlib":
                optf=partial(dlib_optimizer, logger)
            else:
                optf=partial(sceua_optimizer, seed, logger)                
            #smlist.append(GaussianProcessRegressor(kernel=kernel, alpha=1e-5, n_restarts_optimizer=5))
            smlist.append(GaussianProcessRegressor(kernel=kernel, optimizer=optf, normalize_y=True))
            smlist[i].fit(x,y[:,i])
        self.smlist = smlist

    def predict(self,xin):
        x = np.zeros_like(xin)
        if len(x.shape) == 1:
            x = x.reshape((1,self.nInput))
            xin = xin.reshape((1,self.nInput))
        N = x.shape[0]
        y = np.zeros((N, self.nOutput))
        for i in range(N):
            x[i,:] = (xin[i,:] - self.xlb) / self.xrg
        for i in range(self.nOutput):
            y[:,i] = self.smlist[i].predict(x)
        return y

    def evaluate(self,x):
        return self.predict(x)



def dlib_optimizer(logger, obj_func, initial_theta, bounds):
    """
    dlib GFS optimizer for optimizing hyper parameters of GPR
    Input:
      * 'obj_func' is the objective function to be minimized, which
        takes the hyperparameters theta as parameter and an
        optional flag eval_gradient, which determines if the
        gradient is returned additionally to the function value
      * 'initial_theta': the initial value for theta, which can be 
        used by local optimizers
      * 'bounds': the bounds on the values of theta, 
        [(lb1, ub1), (lb2, ub2), (lb3, ub3)]
     Returned:
      * 'theta_opt' is the best found hyperparameters theta
      * 'func_min' is the corresponding value of the target function.
    """
    import dlib
    nopt = len(bounds)
    is_int = []
    lb = []
    ub = []
    for i, bd in enumerate(bounds):
        lb.append(bd[0])
        ub.append(bd[1])
        is_int.append(type(bd[0]) == int and type(bd[1]) == int)
    spec = dlib.function_spec(bound1=lb, bound2=ub, is_integer=is_int)

    progress_frac = 100
    maxn = 1000
    eps = 0.01

    optimizer = dlib.global_function_search([spec])
    optimizer.set_solver_epsilon(eps)

    for i in range(maxn):
        logger.info(f'GPR optimization loop: {i} iterations: ')
        next_eval = optimizer.get_next_x()
        next_eval.set(-obj_func(list(next_eval.x))[0])
        if (i > 0) and (i % progress_frac == 0):
            best_eval = optimizer.get_best_function_eval()
            logger.info(f'GPR optimization loop: {i} iterations: ')
            logger.info(f'  best evaluation so far: {-best_eval[1]} @ {list(best_eval[0])}')

    best_eval = optimizer.get_best_function_eval()
    theta_opt = np.asarray(list(best_eval[0]))
    func_min = -best_eval[1]
    return theta_opt, func_min


    
def sceua_optimizer(seed, logger, obj_func, initial_theta, bounds):
    """
    SCE-UA optimizer for optimizing hyper parameters of GPR
    Input:
      * 'obj_func' is the objective function to be minimized, which
        takes the hyperparameters theta as parameter and an
        optional flag eval_gradient, which determines if the
        gradient is returned additionally to the function value
      * 'initial_theta': the initial value for theta, which can be 
        used by local optimizers
      * 'bounds': the bounds on the values of theta, 
        [(lb1, ub1), (lb2, ub2), (lb3, ub3)]
     Returned:
      * 'theta_opt' is the best found hyperparameters theta
      * 'func_min' is the corresponding value of the target function.
    """
    nopt = len(bounds)
    bl = np.asarray([b[0] for b in bounds])
    bu = np.asarray([b[1] for b in bounds])
    ngs = nopt
    maxn = 3000
    kstop = 10
    pcento = 0.1
    peps = 0.001
    [bestx, bestf, icall, nloop, bestx_list, bestf_list, icall_list] = \
        sceua(obj_func, bl, bu, nopt, ngs, maxn, kstop, pcento, peps, seed=seed, logger=logger)
    theta_opt = bestx
    func_min = bestf
    return theta_opt, func_min



def select_simplex(nps, npg, local_random):
    lcs = set([0])
    for k3 in range(1, nps):
        apos = np.asarray(np.floor(npg + 0.5 - np.sqrt((npg + 0.5)**2 - npg * (npg + 1) * local_random.uniform(size=1000))),
                          dtype=np.uint32)
        for i in range(apos.shape[0]):
            lpos = apos[i]
            if lpos not in lcs:
                break
        lcs.add(lpos)
    return list(lcs)


def sceua(func, bl, bu, nopt, ngs, maxn, kstop, pcento, peps, seed=None, logger=None):
    """
    This is the subroutine implementing the SCE algorithm, 
    written by Q.Duan, 9/2004
    translated to python by gongwei, 11/2017

    Parameters:
    func:   optimized function
    bl:     the lower bound of the parameters
    bu:     the upper bound of the parameters
    nopt:   number of adjustable parameters
    ngs:    number of complexes (sub-populations)
    maxn:   maximum number of function evaluations allowed during optimization
    kstop:  maximum number of evolution loops before convergency
    pcento: the percentage change allowed in kstop loops before convergency
    peps:   relative size of parameter space
    
    npg:  number of members in a complex 
    nps:  number of members in a simplex
    npt:  total number of points in an iteration
    nspl:  number of evolution steps for each complex before shuffling
    mings: minimum number of complexes required during the optimization process

    LIST OF LOCAL VARIABLES
    x[.,.]:    coordinates of points in the population
    xf[.]:     function values of x[.,.]
    xx[.]:     coordinates of a single point in x
    cx[.,.]:   coordinates of points in a complex
    cf[.]:     function values of cx[.,.]
    s[.,.]:    coordinates of points in the current simplex
    sf[.]:     function values of s[.,.]
    bestx[.]:  best point at current shuffling loop
    bestf:     function value of bestx[.]
    worstx[.]: worst point at current shuffling loop
    worstf:    function value of worstx[.]
    xnstd[.]:  standard deviation of parameters in the population
    gnrng:     normalized geometri%mean of parameter ranges
    lcs[.]:    indices locating position of s[.,.] in x[.,.]
    bound[.]:  bound on ith variable being optimized
    ngs1:      number of complexes in current population
    ngs2:      number of complexes in last population
    criter[.]: vector containing the best criterion values of the last 10 shuffling loops
    """

    verbose = (logger is not None)
    local_random = np.random.default_rng(seed=seed)
    
    # Initialize SCE parameters:
    npg  = 2 * nopt + 1
    nps  = nopt + 1
    nspl = npg
    npt  = npg * ngs
    bd   = bu - bl

    # Create an initial population to fill array x[npt,nopt]
    x_sample = local_random.uniform(size=(npt, nopt))
    x = x_sample * bd + bl

    xf = np.zeros(npt)
    for i in range(npt):
        xf[i] = func(x[i,:])[0] # only used the first returned value
    
    icall = npt

    # Sort the population in order of increasing function values
    idx = np.argsort(xf)
    xf = xf[idx]
    x = x[idx,:]

    # Record the best and worst points
    bestx  = np.copy(x[0,:])
    bestf  = np.copy(xf[0])
    worstx = np.copy(x[-1,:])
    worstf = np.copy(xf[-1])
    
    bestf_list = []
    bestf_list.append(bestf)
    bestx_list = []
    bestx_list.append(bestx)
    icall_list = []
    icall_list.append(icall)
    
    if verbose:
        logger.info('The Initial Loop: 0')
        logger.info(f'BESTF  : {bestf:.2f}')
        logger.info(f'BESTX  : {np.array2string(bestx)}')
        logger.info(f'WORSTF : {worstf:.2f}')
        logger.info(f'WORSTX : {np.array2string(worstx)}')
        logger.info(' ')

    # Computes the normalized geometric range of the parameters
    gnrng = np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bd)))
    # Check for convergency
    if verbose:
        if icall >= maxn:
            logger.info('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
            logger.info('ON THE MAXIMUM NUMBER OF TRIALS ')
            logger.info(maxn)
            logger.info('HAS BEEN EXCEEDED.  SEARCH WAS STOPPED AT TRIAL NUMBER:')
            logger.info(icall)
            logger.info('OF THE INITIAL LOOP!')

        if gnrng < peps:
            logger.info('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')
    
    # Begin evolution loops:
    nloop = 0
    criter = []
    criter_change = 1e+5
    cx = np.zeros([npg,nopt])
    cf = np.zeros(npg)

    while (icall < maxn) and (gnrng > peps) and (criter_change > pcento):
        nloop += 1
        
        # Loop on complexes (sub-populations)
        for igs in range(ngs):

            # Partition the population into complexes (sub-populations)
            k1 = np.int64(np.linspace(0, npg-1, npg))
            k2 = k1 * ngs + igs
            cx[k1,:] = np.copy(x[k2,:])
            cf[k1] = np.copy(xf[k2])
            
            # Evolve sub-population igs for nspl steps
            for loop in range(nspl):
                
                # Select simplex by sampling the complex according to a linear
                # probability distribution
                lcs = np.asarray(select_simplex(nps, npg, local_random))
    
                # Construct the simplex:
                s = np.copy(cx[lcs,:])
                sf = np.copy(cf[lcs])
                
                snew, fnew, icall = cceua(func, s, sf, bl, bu, icall, local_random)
    
                # Replace the worst point in Simplex with the new point:
                s[nps-1,:] = snew
                sf[nps-1] = fnew
                
                # Replace the simplex into the complex
                cx[lcs,:] = np.copy(s)
                cf[lcs] = np.copy(sf)
                
                # Sort the complex
                idx = np.argsort(cf)
                cf = cf[idx]
                cx = cx[idx,:]
                
            # End of Inner Loop for Competitive Evolution of Simplexes
    
            # Replace the complex back into the population
            x[k2,:] = np.copy(cx[k1,:])
            xf[k2] = np.copy(cf[k1])
        
        # End of Loop on Complex Evolution;
        
        # Shuffled the complexes
        idx = np.argsort(xf)
        xf = xf[idx]
        x = x[idx,:]
        
        # Record the best and worst points
        bestx  = np.copy(x[0,:])
        bestf  = np.copy(xf[0])
        worstx = np.copy(x[-1,:])
        worstf = np.copy(xf[-1])
        bestf_list.append(bestf)
        bestx_list.append(bestx)
        icall_list.append(icall)
        
        if verbose:
            logger.info(f'Evolution Loop: {nloop} - Trial - {icall}')
            logger.info(f'BESTF  : {bestf:.2f}')
            logger.info(f'BESTX  : {np.array2string(bestx)}')
            logger.info(f'WORSTF : {worstf:.2f}')
            logger.info(f'WORSTX : {np.array2string(worstx)}')
            logger.info(' ')
        
        # Computes the normalized geometric range of the parameters
        gnrng = np.exp(np.mean(np.log((np.max(x,axis=0)-np.min(x,axis=0))/bd)))
        # Check for convergency;
        if verbose:
            if icall >= maxn:
                logger.info('*** OPTIMIZATION SEARCH TERMINATED BECAUSE THE LIMIT')
                logger.info(f'ON THE MAXIMUM NUMBER OF TRIALS {maxn} HAS BEEN EXCEEDED!')
            if gnrng < peps:
                logger.info('THE POPULATION HAS CONVERGED TO A PRESPECIFIED SMALL PARAMETER SPACE')
            
    
        criter.append(bestf)
        if nloop >= kstop:
            criter_change = np.abs(criter[nloop-1] - criter[nloop-kstop])*100
            criter_change /= np.mean(np.abs(criter[nloop-kstop:nloop]))
            if criter_change < pcento:
                if verbose:
                    logger.info(f'THE BEST POINT HAS IMPROVED IN LAST {kstop} LOOPS BY LESS THAN THE THRESHOLD {pcento}%')
                    logger.info('CONVERGENCY HAS ACHIEVED BASED ON OBJECTIVE FUNCTION CRITERIA!!!')
        
    # End of the Outer Loops
    
    if verbose:
        logger.info(f'SEARCH WAS STOPPED AT TRIAL NUMBER: {icall}')
        logger.info(f'NORMALIZED GEOMETRIC RANGE = {gnrng}')
        logger.info(f'THE BEST POINT HAS IMPROVED IN LAST {kstop} LOOPS BY {criter_change:.4f}%')
   
    # END of Subroutine SCEUA_runner
    return bestx, bestf, icall, nloop, bestx_list, bestf_list, icall_list


def cceua(func, s, sf, bl, bu, icall, local_random):
    """
    This is the subroutine for generating a new point in a simplex
    func:   optimized function
    s[.,.]: the sorted simplex in order of increasing function values
    sf[.]:  function values in increasing order

    LIST OF LOCAL VARIABLES
    sb[.]:   the best point of the simplex
    sw[.]:   the worst point of the simplex
    w2[.]:   the second worst point of the simplex
    fw:      function value of the worst point
    ce[.]:   the centroid of the simplex excluding wo
    snew[.]: new point generated from the simplex
    iviol:   flag indicating if constraints are violated
             = 1 , yes
             = 0 , no
    """

    
    
    nps, nopt = s.shape
    n = nps
    alpha = 1.0
    beta = 0.5

    # Assign the best and worst points:
    sw = s[-1,:]
    fw = sf[-1]

    # Compute the centroid of the simplex excluding the worst point:
    ce = np.mean(s[:n-1,:],axis=0)

    # Attempt a reflection point
    snew = ce + alpha * (ce - sw)

    # Check if is outside the bounds:
    ibound = 0
    s1 = snew - bl
    if sum(s1 < 0) > 0:
        ibound = 1
    s1 = bu - snew
    if sum(s1 < 0) > 0:
        ibound = 2
    if ibound >= 1:
        snew = bl + local_random.random(nopt) * (bu - bl)

    fnew = func(snew)[0] # only used the first returned value
    icall += 1

    # Reflection failed; now attempt a contraction point
    if fnew > fw:
        snew = sw + beta * (ce - sw)
        fnew = func(snew)[0] # only used the first returned value
        icall += 1
    
    # Both reflection and contraction have failed, attempt a random point
        if fnew > fw:
            snew = bl + local_random.random(nopt) * (bu - bl)
            fnew = func(snew)[0] # only used the first returned value
            icall += 1

    # END OF CCE
    return snew, fnew, icall
