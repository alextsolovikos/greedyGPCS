#!/usr/bin/env python

import numpy as np
from scipy import sparse
import time
from scipy import linalg
from matplotlib import pyplot as plt
import argparse
import cvxpy as cp


"""
Functions for solving the linearized mean and covariance steering problem.
"""

def linearCovarianceSteeringDiscrete(A, B, drift, mu_0, Sigma_0, mu_f, Sigma_f, W, N, previous_soln=None, verbose=False):
    """
        Optimal mean and covariance steering of a discrete-time LTI system.
        Based on the paper: "Greedy Finite-Horizon Covariance Steering for 
        Discrete-Time Stochastic Nonlinear Systems Based on the Unscented 
        Transform" by E. Bakolas and A. Tsolovikos

        Inputs:
            A:          State matrix: (nx,nx) if A is constant, otherwise (N+1,nx,nx)
            B:          Input matrix ((A,B) must be controllable): (nx,nu) if B is constant otherwise (N+1,nx,nu)
	    d:          Drift term
	    mu_0:       Initial mean
	    Sigma_0:    Initial covariance
	    mu_f:       Target mean
	    Sigma_f:    Target covariance
            N:          Number of discrete-time steps 
                        (Initial time step: k = 0; Terminal time step: k = N)

	Outputs:
	    u_star:     Constant part of optimal input
	    K_star:     Feedback part of optimal input

    """
    
    if A.ndim == 3:
        if verbose:
            print('The linear dynamics are time-varying')
        time_invariant = False
        if A.shape[0] != N:
            raise ValueError('For time-varying matrix A, the shape of the input A has to be: (N,nx,nx).')
        if B.shape[0] != N:
            raise ValueError('For time-varying matrix B, the shape of the input B has to be: (N,nx,nu).')
        if drift.shape[0] != N:
            raise ValueError('For time-varying drift, the shape of the input drift has to be: (N,nx).')
    elif A.ndim == 2:
        if verbose:
            print('The linear dynamics are time-invariant')
        time_invariant = True
    else:
        raise ValueError('A matrix needs to have either 2 (time-invariant) or 3 (time-varying) dimensions. The given matrix has: A.ndim = ', A.ndim)

    if not time_invariant and B.ndim == 2:
        raise ValueError('For time-invariant linear dynamics, B must be of shape (nx,nu)')
    elif time_invariant and B.ndim == 1:
        raise ValueError('For time-varying linear dynamics, B must be of shape (N,nx,nu)')

    def Phi(k, n):
        if k == n:
            return np.eye(nx)
        if k < n:
            raise ValueError('In Phi(k,n): k has to be greater that or equal to n.')
        if time_invariant:
            return np.linalg.matrix_power(A, k-n)
        else:
            Product = A[n]
            for i in range(n+1,k):
                Product = A[i] @ Product
            return Product

    # Basic parameters
    if time_invariant:
        nx = A.shape[0] 
        nu = B.shape[1] # Number of inputs
    else:
        nx = A.shape[1]
        nu = B.shape[2] # Number of inputs

    if verbose:
        print('Number of states: ', nx)
        print('Number of inputs: ', nu)

    # TO DO: Add fallbacks for wrong matrix sizes

    # Initialize fundamental matrices
    Gu = np.zeros(((N + 1) * nx, N * nu))
    Gw = np.zeros(((N + 1) * nx, N * nx))
    Gz = np.zeros(((N + 1) * nx, nx))

    if time_invariant:
        d = np.tile(drift, N)
    else:
        d = drift.flatten()
    WW = np.kron(np.eye(N, dtype=int), W)
    v = np.zeros((N * nu))  # This is the open-loop input at every time step
    K = np.zeros((N * nu, (N + 1) * nx))    # This is the feedback part of the input

    t = time.time()
    if verbose:
        print('Setting up SDP...')
    for i in range(N + 1):
        Gz[i*nx:(i+1)*nx] = Phi(i,0)
        for j in range(i):
            Gw[i*nx:(i+1)*nx, j*nx:(j+1)*nx] =  Phi(i,j+1)
            if time_invariant:
                Gu[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Phi(i,j+1) @ B
            else:
                Gu[i*nx:(i+1)*nx, j*nu:(j+1)*nu] = Phi(i,j+1) @ B[j]

    # Useful matrices for constraints
    S1 = Gz @ Sigma_0 @ Gz.T
    S2 = Gz @ mu_0
    S3 = Gw @ d
    S4 = Gw @ WW @ Gw.T

    R11 = S1 + np.outer(S2, S2) + np.outer(S2, S3) + np.outer(S2, S3).T + S4 + np.outer(S3, S3)
    R12 = (S2 + S3).reshape(-1,1)
    R21 = (S2 + S3).reshape(1,-1)
    R22 = 1.

    Q = np.block([[R11, R12], [R21, R22]])

    PN = np.hstack((np.zeros((nx, N * nx)), np.eye(nx)))
    Gu_til = PN @ Gu
    R11_til = PN @ R11
    R12_til = (PN @ R12)
    R11_bar = PN @ R11 @ PN.T

    # Initialize Y if previous solution is avaliable
    if previous_soln is None:
        Y_prev = np.zeros((N * nu, (N + 1) * nx + 1))
    else:
        u_prev = np.zeros(N * nu)
        K_prev = np.zeros((N * nu, (N + 1) * nx))
        for k in range(N):
            u_prev[k*nu:(k+1)*nu] = previous_soln[0][k]
            for j in range(k+1):
                K_prev[k*nu:(k+1)*nu, j*nx:(j+1)*nx] = previous_soln[1][k,j]

        L_prev = K_prev @ np.linalg.inv(np.eye((N+1) * nx) - Gu @ K_prev)
        v_prev = (np.eye(N * nu) + L_prev @ Gu) @ u_prev
        Y_prev = np.vstack((L_prev, v_prev))


    # Make left part of Y block lower diagonal
    y_zero_indices = []
    for i in range(N):
        for j in range(nu):
            for k in range((i+1)*nx,(N+1)*nx):
                y_zero_indices.append((i*nu+j,k))

    # Solve Convex Program
#   Y = cp.Variable((N * nu, (N + 1) * nx + 1), value=Y_prev)
    Y = cp.Variable((N * nu, (N + 1) * nx + 1))
    # Mean constraint
    constraints = [R12_til + Gu_til @ Y @ np.vstack((R12, [1.])) - mu_f.reshape(-1,1) == 0]
    # Block lower-triangular constraint
    for idx in y_zero_indices:
        constraints += [Y[idx] == 0]

    # Covariance semidefinite constraint
    Sigma_til = Sigma_f + np.outer(mu_f, mu_f) - (R11_bar + Gu_til @ Y @ np.vstack((R11_til.T, R12_til.T)) + np.hstack((R11_til, R12_til)) @ Y.T @ Gu_til.T)
    Q_til = np.linalg.cholesky(Q)
    Y_til = Gu_til @ Y @ Q_til
    constraints += [cp.bmat([[Sigma_til, Y_til], [Y_til.T, np.eye((N + 1) * nx + 1)]]) >> 0]

    # Define and solve problem
    prob = cp.Problem(cp.Minimize(cp.sum([cp.quad_form(Y[i], Q) for i in range(N * nu)])), constraints)
    if verbose:
        print('Solving SDP...')
    prob.solve(
        solver=cp.MOSEK, 
        verbose=verbose, 
    )  

    # If not solved using MOSEK
    if Y.value is None:
        print('MOSEK failed to find a solution. Trying SCS...')
        Y = cp.Variable((N * nu, (N + 1) * nx + 1))
        # Mean constraint
        constraints = [R12_til + Gu_til @ Y @ np.vstack((R12, [1.])) - mu_f.reshape(-1,1) == 0]
        # Block lower-triangular constraint
        for idx in y_zero_indices:
            constraints += [Y[idx] == 0]
        # Covariance semidefinite constraint
        Sigma_til = Sigma_f + np.outer(mu_f, mu_f) - (R11_bar + Gu_til @ Y @ np.vstack((R11_til.T, R12_til.T)) + np.hstack((R11_til, R12_til)) @ Y.T @ Gu_til.T)
        Q_til = np.linalg.cholesky(Q)
        Y_til = Gu_til @ Y @ Q_til
        constraints += [cp.bmat([[Sigma_til, Y_til], [Y_til.T, np.eye((N + 1) * nx + 1)]]) >> 0]
        prob = cp.Problem(cp.Minimize(cp.sum([cp.quad_form(Y[i], Q) for i in range(N * nu)])), constraints)
        prob.solve(
            solver=cp.SCS, 
            verbose=True, 
            alpha=0.01, 
            use_indirect=False,
            max_iters=60000, 
            normalize=False,
            eps=1.e-1,
            warm_start=False,
        )  
        
    Y_opt = Y.value

    L = Y_opt[:,:-1]
    neu = Y_opt[:,-1]
    LK = np.linalg.inv(np.eye(N * nu) + L @ Gu) 

    K = LK @ L
    v = LK @ neu

    KK = np.zeros((N, N, nu, nx))
    vv = np.zeros((N, nu))
    for k in range(N):
        vv[k] = v[k*nu:(k+1)*nu]
        for j in range(k+1):
            KK[k,j] = K[k*nu:(k+1)*nu, j*nx:(j+1)*nx]

    if verbose:
        print('Optimal value:')
        print(prob.value)
        print(f'SDP solved in {time.time() - t} s')
    
    return vv, KK


def unscented_transform_with_feedback(mu, P, u, K, W, f, u_corr=None):

    # Parameters
    n = mu.shape[0]
    alpha = 0.5
    beta = 2.0
    lamb = alpha ** 2 * n - n

    # Coefficients gamma & delta
    gamma = np.zeros((2 * n + 1, 1))
    delta = np.zeros((2 * n + 1, 1))

    for i in range(2 * n + 1):
        gamma[i] = lamb / (lamb + n) if i == 0 else 1. / (2 * (lamb + n))
        delta[i] = 1. - alpha ** 2 + beta + lamb / (lamb + n) if i == 0 else 1. / (2. * (lamb + n))

    # Sigma points
    sigma = np.zeros((n, 2 * n + 1))
    Psqrt = linalg.sqrtm(P)

    for i in range(2 * n + 1):
        if i == 0:
            sigma[:,i] = mu
        elif i <= n:
            sigma[:,i] = mu + np.sqrt(n + lamb) * Psqrt[:,i - 1]
        else:
            sigma[:,i] = mu - np.sqrt(n + lamb) * Psqrt[:,i - n - 1]

    # Propagate sigma points
    sigma_hat = np.zeros((n, 2 * n + 1))
    for i in range(2 * n + 1):
        if u_corr is None:
            u_sigma = u + K @ sigma[:,i]
        else:
            u_sigma = u + K @ sigma[:,i] + u_corr[:,i]
        sigma_hat[:,i] = f(sigma[:,i], u_sigma)

    # Mean and covariance
    mu_hat = (sigma_hat @ gamma).flatten()
    P_hat = W
    for i in range(2 * n + 1):
        P_hat += delta[i] * np.outer(sigma_hat[:,i] - mu_hat, sigma_hat[:,i] - mu_hat)
    

    # Note: Process noise covariance is defined in assumptions.py

    return mu_hat, P_hat, sigma.T, sigma_hat.T



"""
Utilities
"""

def get_covariance_at_angle(a=1., b=1., theta=0.):
    return np.array([[a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2, (a**2 - b**2) * np.sin(theta) * np.cos(theta)],
                     [(a**2 - b**2) * np.sin(theta) * np.cos(theta), a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2]
                    ])

