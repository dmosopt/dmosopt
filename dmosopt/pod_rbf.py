"""
An implementation of Proper Orthogonal Decomposition - Radial Basis Function (POD-RBF) method.

Based on code by Kyle Beggs: https://github.com/UCF-ERAU-OH-Research-Group/POD-RBF
"""
import numpy as np


class POD_RBF:
    def __init__(
        self,
        xin,
        yin,
        nInput,
        nOutput,
        xlb,
        xub,
        mem_limit=16 * 1024**3,
        energy_threshold=0.99,
        logger=None,
    ):

        self.mem_limit = mem_limit
        self.energy_threshold = energy_threshold
        self.snapshot = None

        self.nInput = nInput
        self.nOutput = nOutput
        self.xlb = xlb
        self.xub = xub
        self.xrg = xub - xlb
        self.logger = logger

        N = xin.shape[0]
        x = np.zeros_like(xin)
        y = np.copy(yin)
        for i in range(N):
            x[i, :] = (xin[i, :] - self.xlb) / self.xrg
        if nOutput == 1:
            y = y.reshape((y.shape[0], 1))

        self.train(x.T, y.T)

    def _calc_truncated_POD_basis(self):
        """
        Calculate the truncated POD basis.

        Parameters
        ----------
        snapshot : ndarray
            The matrix containing data points for each parameter as columns.
        energyThreshold : float
            The minimum percent of energy to keep in the system.

        Returns
        -------
        ndarray
            The truncated POD basis.

        """

        # calculate the memory in gigabytes
        memory = self.snapshot.nbytes

        if memory < self.mem_limit:
            # calculate the SVD of the snapshot
            U, S, _ = np.linalg.svd(self.snapshot, full_matrices=False)

            # calculate the truncated POD basis based on the amount of energy/POD modes required
            self.cumulative_energy = np.cumsum(S) / np.sum(S)
            if self.energy_threshold >= 1:
                self.truncated_energy = 1
                trunc_id = len(S)
                return U
            elif self.energy_threshold < self.cumulative_energy[0]:
                trunc_id = 0
            else:
                trunc_id = np.argmax(self.cumulative_energy > self.energy_threshold)

            self.truncated_energy = self.cumulative_energy[trunc_id]
            basis = U[:, : (trunc_id + 1)]
        else:
            # compute the covariance matrix and corresponding eigenvalues and eigenvectors
            cov = np.matmul(np.transpose(self.snapshot), self.snapshot)
            self.eig_vals, self.eig_vecs = np.linalg.eigh(cov)
            self.eig_vals = np.abs(self.eig_vals.real)
            self.eig_vecs = self.eig_vecs.real
            # rearrange eigenvalues and eigenvectors from largest -> smallest
            self.eig_vals = self.eig_vals[::-1]
            self.eig_vecs = self.eig_vecs[:, ::-1]

            # calculate the truncated POD basis based on the amount of energy/POD modes required
            self.cumulative_energy = np.cumsum(self.eig_vals) / np.sum(self.eig_vals)
            if self.energy_threshold >= 1:
                self.truncated_cumulative_energy = 1
                trunc_id = len(self.eig_vals)
            elif self.energy_threshold < self.cumulative_energy[0]:
                trunc_id = 1
            else:
                trunc_id = np.argmax(self.cumulative_energy > self.energy_threshold)

            self.truncated_energy = self.cumulative_energy[trunc_id]
            self.eig_vals = self.eig_vals[: (trunc_id + 1)]
            self.eig_vecs = self.eig_vecs[:, : (trunc_id + 1)]

            # calculate the truncated POD basis
            basis = (self.snapshot @ self.eig_vecs) / np.sqrt(self.eig_vals)

        return basis

    def _build_collocation_matrix(self, c):
        num_train_points = self.train_params.shape[1]
        num_params = self.train_params.shape[0]
        r2 = np.zeros((num_train_points, num_train_points))
        for i in range(num_params):
            I, J = np.meshgrid(
                self.train_params[i, :],
                self.train_params[i, :],
                indexing="ij",
                copy=False,
            )
            r2 += ((I - J) / self.params_range[i]) ** 2
        return 1 / np.sqrt(r2 / (c**2) + 1)

    def _find_optim_shape_param(self, cond_range=[1e11, 1e12], max_steps=1e5):
        optim_c = 1
        found_optim_c = False
        k = 0
        while found_optim_c is False:
            k += 1
            if optim_c < 0:
                ValueError("Shape parameter is negative.")
            C = self._build_collocation_matrix(optim_c)
            cond = np.linalg.cond(C)
            if cond <= cond_range[0]:
                optim_c += 0.01
            if cond > cond_range[1]:
                optim_c -= 0.01
            if cond > cond_range[0] and cond < cond_range[1]:
                found_optim_c = True
            if optim_c < 0.011:
                raise ValueError(
                    "Shape factor cannot be less than 0. shape_factor={}".format(
                        optim_c
                    )
                )
            if k > max_steps:
                print("WARNING: MAX STEPS")
                break

        return optim_c

    def _build_RBF_inference_matrix(self, params):
        """Make the Radial Basis Function (RBF) matrix using the
         Hardy Inverse Multi-Qualdrics (IMQ) function

        Parameters
        ----------
        params : ndarray
            The parameters to inference the RBF network on.

        Returns
        -------
        ndarray
            The RBF matrix.

        """

        params = np.transpose(params)

        assert params.shape[0] == self.train_params.shape[0]

        num_params = self.train_params.shape[0]
        num_train_points = self.train_params.shape[1]
        r2 = np.zeros((num_params, num_train_points))
        for i in range(num_params):
            I, J = np.meshgrid(
                params[i, :],
                self.train_params[i, :],
                indexing="ij",
                copy=False,
            )
            r2 += ((I - J) / self.params_range[i]) ** 2
        return 1 / np.sqrt(r2 / (self.shape_factor**2) + 1)

    def train(self, train_params, snapshot, shape_factor=None):
        """
        Train the Radial Basis Function (RBF) network

        Parameters
        ----------
        train_params : ndarray
            The parameters used to generate the snapshot matrix.
        snapshot : ndarray
            The matrix containing data points for each parameter as columns.
        shape_factor : float
            The shape factor to be used in the RBF network.

        Returns
        -------
        ndarray
            The weights/coefficients of the RBF network.

        """
        if train_params.ndim < 2:
            assert (
                snapshot.shape[1] == train_params.shape[0]
            ), "Number of parameter points ({}) and snapshots ({}) not the same".format(
                train_params.shape[1], snapshot.shape[1]
            )
            self.params_range = np.array([np.ptp(train_params, axis=0)])
        else:
            assert (
                snapshot.shape[1] == train_params.shape[1]
            ), "Number of parameter points ({}) and snapshots ({}) not the same".format(
                train_params.shape[1], snapshot.shape[1]
            )
            self.params_range = np.ptp(train_params, axis=1)
        self.snapshot = snapshot
        if train_params.ndim < 2:
            self.train_params = np.expand_dims(train_params, axis=0)
        else:
            self.train_params = train_params

        if shape_factor is None:
            self.shape_factor = self._find_optim_shape_param()
        else:
            self.shape_factor = shape_factor
        self.basis = self._calc_truncated_POD_basis()

        # build the Radial Basis Function (RBF) matrix
        F = self._build_collocation_matrix(self.shape_factor)

        # calculate the amplitudes (A) and weights/coefficients (B)
        A = np.matmul(np.transpose(self.basis), self.snapshot)
        self.weights = A @ np.linalg.pinv(F.T)

    def predict(self, xin):
        """Interpolate an observation with the RBF network.

        Parameters
        ----------
        xin : ndarray
            The parameters to interpolate with the RBF network.

        Returns
        -------
        ndarray
            The output of the RBF network according to the xin argument.

        """

        x = np.zeros_like(xin)
        if len(x.shape) == 1:
            x = x.reshape((1, self.nInput))
            xin = xin.reshape((1, self.nInput))

        N = x.shape[0]
        for i in range(N):
            x[i, :] = (xin[i, :] - self.xlb) / self.xrg

        # build the Radial Basis Function (RBF) matrix
        F = self._build_RBF_inference_matrix(x)

        # calculate the inferenced solution
        A = np.matmul(self.weights, np.transpose(F))
        result = np.matmul(self.basis, A)

        return result[:, 0]

    def evaluate(self, xin):
        return self.predict(xin)
