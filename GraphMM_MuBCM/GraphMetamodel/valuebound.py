'''
@Author: Angdi
@Date:   2022/6/6
@File    : boundmodel.py
@Email   : liad@shanghaitech.edu.cn

To add lower/upper bound in UKF.
use it as:
bound_UKF(lowerbound = a, upperbound = b)

'''


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from IPython.display import display, Math
import random


from filterpy.kalman import unscented_transform
from numpy import eye, zeros, dot, isscalar, outer
from copy import deepcopy
from scipy.linalg import cholesky, sqrtm


class bound_MerweScaledSigmaPoints(MerweScaledSigmaPoints):
    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        super().__init__(n, alpha, beta, kappa, sqrt_method, subtract)
        

        if sqrt_method is None:
            # self.sqrt = cholesky
            self.sqrt   = sqrtm
        else:
            self.sqrt = sqrt_method


    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.

        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I

        Parameters
        ----------

        x : An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])

        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.

        Returns
        -------

        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.

            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """

        if self.n != np.size(x):
            raise ValueError("expected size(x) {}, but size is {}".format(
                self.n, np.size(x)))

        n = self.n

        if np.isscalar(x):
            x = np.asarray([x])

        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.atleast_2d(P)

        lambda_ = self.alpha**2 * (n + self.kappa) - n
        
        nn = (lambda_ + n)*P
        # for i in range(len(nn)):
        #     print(nn[i])
        
        U = self.sqrt((lambda_ + n)*P)

        sigmas = np.zeros((2*n+1, n))
        sigmas[0] = x
        for k in range(n):
            # pylint: disable=bad-whitespace
            sigmas[k+1]   = self.subtract(x, -U[k])
            sigmas[n+k+1] = self.subtract(x, U[k])

        return sigmas




class bound_UKF(UKF):
    def __init__(self, dim_x, dim_z, dt, hx, fx, points, sqrt_fn=None, x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None, lowerbound = None, upperbound = None):
        super().__init__(dim_x, dim_z, dt, hx, fx, points, sqrt_fn, x_mean_fn, z_mean_fn, residual_x, residual_z)
        self.lowerboud = lowerbound
        self.upperbound = upperbound

    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        r"""
        Performs the predict step of the UKF. On return, self.x and
        self.P contain the predicted state (x) and covariance (P). '

        Important: this MUST be called before update() is called for the first
        time.

        Parameters
        ----------

        dt : double, optional
            If specified, the time step to be used for this prediction.
            self._dt is used if this is not provided.

        fx : callable f(x, **fx_args), optional
            State transition function. If not provided, the default
            function passed in during construction will be used.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **fx_args : keyword arguments
            optional keyword arguments to be passed into f(x).
        """

        if dt is None:
            dt = self._dt

        if UT is None:
            UT = unscented_transform

        # calculate sigma points for given mean and covariance
        self.compute_process_sigmas(dt, fx, **fx_args)

        #and pass sigmas through the unscented transform to compute prior
        self.x, self.P = UT(self.sigmas_f, self.Wm, self.Wc, self.Q,
                            self.x_mean, self.residual_x)

        # converge matrix 
        self.x[np.isnan(self.x)] = 0


        # save prior
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)


    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        """
        add lower boundary to for x 
        ----------

        Update the UKF with the given measurements. On return,
        self.x and self.P contain the new mean and covariance of the filter.

        Parameters
        ----------

        z : numpy.array of shape (dim_z)
            measurement vector

        R : numpy.array((dim_z, dim_z)), optional
            Measurement noise. If provided, overrides self.R for
            this function call.

        UT : function(sigmas, Wm, Wc, noise_cov), optional
            Optional function to compute the unscented transform for the sigma
            points passed through hx. Typically the default function will
            work - you can use x_mean_fn and z_mean_fn to alter the behavior
            of the unscented transform.

        **hx_args : keyword argument
            arguments to be passed into h(x) after x -> h(x, **hx_args)
        """
        # print(111)
        if z is None:
            self.z = np.array([[None]*self._dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if hx is None:
            hx = self.hx

        if UT is None:
            UT = unscented_transform

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self._dim_z) * R

        # pass prior sigmas through h(x) to get measurement sigmas
        # the shape of sigmas_h will vary if the shape of z varies, so
        # recreate each time
        sigmas_h = []
        for s in self.sigmas_f:
            sigmas_h.append(hx(s, **hx_args))

        self.sigmas_h = np.atleast_2d(sigmas_h)

        # mean and covariance of prediction passed through unscented transform
        zp, self.S = UT(self.sigmas_h, self.Wm, self.Wc, R, self.z_mean, self.residual_z)
        self.SI = self.inv(self.S)

        # compute cross variance of the state and the measurements
        Pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)


        self.K = dot(Pxz, self.SI)        # Kalman gain
        self.y = self.residual_z(z, zp)   # residual

        # update Gaussian state estimate (x, P)
        self.x = self.x + dot(self.K, self.y)  ## x mean

        ## set boundary

        # testvalue = [x < self.lower_bound for x in self.x ]
        # print(11, testvalue, '\n')
        if self.lowerboud != None:
            for ii, value in enumerate([x < self.lowerboud for x in self.x ]):
                if value: self.x[ii] = self.lowerboud
        # print( self.x)

        # upper bound
        if self.upperbound != None:
            for ii, value in enumerate([x > self.upperbound for x in self.x ]):
                if value: self.x[ii] = self.upperbound
        
        
        self.P = self.P - dot(self.K, dot(self.S, self.K.T))

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

