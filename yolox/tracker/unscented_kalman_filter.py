import numpy as np
import scipy.linalg
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}

class UnscentedKalmanFilter(object):
    def __init__(self):
        # Khởi tạo các tham số
        ndim = 4  # Kích thước của vectơ trạng thái
        dt = 1.  # Khoảng thời gian giữa các bước

        # Khởi tạo ma trận chuyển động
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Khởi tạo trọng số của trạng thái và vận tốc
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def _unscented_transform(self, sigmas, Wm, Wc, noise_cov):
        """ Unscented transform step.
        
        Parameters:
        -----------
        sigmas : ndarray
            sigma points
        Wm : ndarray
            Weights for the mean
        Wc : ndarray
            Weights for the covariance
        noise_cov : ndarray
            Covariance matrix for additive noise
            
        Returns:
        --------
        ndarray
            Mean of the transformed points
        ndarray
            Covariance matrix of the transformed points
        """
        mean = np.dot(Wm, sigmas)
        residual = sigmas - mean[None, :]
        cov = np.dot(residual.T, np.dot(np.diag(Wc), residual))
        cov += noise_cov
        return mean, cov

    def predict(self, mean, covariance):
        # Xác định ma trận nhiễu chuyển động
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        wm_0, wm_i, wc_0, wc_i = self.compute_weights(n=8)
        points = self.sigma_points(mean, covariance, 1.0)
        predicted_points = np.dot(points, self._motion_mat.T)
        print(predicted_points.shape)
        predicted_mean, predicted_cov = self.unscented_transform(
            predicted_points, wm_i, wc_i, motion_cov)

        return predicted_mean, predicted_cov

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self.update_mat, mean)
        covariance = np.linalg.multi_dot((
            self.update_mat, covariance, self.update_mat.T))
        return mean, covariance + innovation_cov
    
    def unscented_transform(self, sigmas, Wm, Wc, noise_cov):
        mean = np.dot(Wm, sigmas)
        residual = sigmas - mean[None, :]
        cov = np.dot(residual.T, Wc * residual)
        cov += noise_cov
        return mean, cov
    
    def _multi_predict(self, mean, covariance):
        """Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrics of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        # UKF
        dt = 0.75
        k1 = 9.8
        k2 = 9.8
        motion_cov = []
        for i in range(len(mean)):
            self._motion_mat[0][4] = dt
            self._motion_mat[1][5] = dt
            self._motion_mat[2][3] = k1 * mean[i][6] * dt
            self._motion_mat[3][7] = k2 * mean[i][7] * dt

            # Xác định các siêu điểm
            points = MerweScaledSigmaPoints(8, alpha=.1, beta=2., kappa=-1)
            sigmas = points.sigma_points(mean[i], np.diag(sqr[i]))
            
            # points = UKF.sigma_points(mean[i], covariance[i], 1.0)

            # Xấp xỉ phân phối hậu nghiệm
            mean[i], covariance[i] = unscented_transform(
                mean[i], points.Wm, points.Wc)

            motion_cov.append(np.diag(sqr[i]))

        motion_cov = np.asarray(motion_cov)
        #mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance
    
    
    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        """Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')
        
    def sigma_points(self, mean, covariance, kappa):
        n = len(mean)
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        sqrt_cov = np.linalg.cholesky((n + kappa) * covariance)
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_cov[i]
            sigma_points[n + i + 1] = mean - sqrt_cov[i]
        return sigma_points
    
    def compute_weights(self, n, alpha=1e-3, beta=2, kappa=0):
        lam = alpha**2 * (n + kappa) - n
        wm_0 = lam / (n + lam)
        wm_i = 1 / (2 * (n + lam))
        wc_0 = wm_0 + (1 - alpha**2 + beta)
        wc_i = wm_i
        return wm_0, wm_i, wc_0, wc_i
    
    def multi_predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        # Calculate sigma points
        temp_mean = []
        temp_cov = []
        for i in range(len(mean)):
            points = self.sigma_points(mean[i], covariance[i], 1.0)

            # Predict sigma points
            predicted_points = np.dot(points, self._motion_mat.T)

            # Calculate motion covariance matrix
            motion_cov = np.array([np.diag(s) for s in sqr])

            # Perform unscented transform for each sigma point
            predicted_means = []
            predicted_covs = []
            wm_0, wm_i, wc_0, wc_i = self.compute_weights(n=8)
            for i in range(len(predicted_points)):
                mean, cov = self.unscented_transform(
                    predicted_points[i], wm_i, wc_i, motion_cov[i])
                predicted_means.append(mean)
                predicted_covs.append(cov)
            temp_mean.append(predicted_means)
            temp_cov.append(predicted_covs)
        print(temp_mean.shape)
        print(temp_cov.shape)

        return np.array(predicted_means), np.array(predicted_covs)


