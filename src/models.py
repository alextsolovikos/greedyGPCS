import numpy as np

class SimpleCar():
    """
    Simple 2D car nonlinear dynamical model (nx = 4, nu = 2)

    States:
        x[k,0] -> coordinate x
        x[k,1] -> coordinate y
        x[k,2] -> heading theta
        x[k,3] -> velocity u

    Inputs:
        u[k,0] -> control on steering angle
        u[k,1] -> control on velocity
        
    Dynamics:
        x[k+1,0] = x[k,0] + x[k,3] * cos(x[k,2]) * dt   -> coordinate x
        x[k+1,1] = x[k,1] + x[k,3] * sin(x[k,2]) * dt   -> coordinate y
        x[k+1,2] = x[k,2] + u[k,0] * x[k,3] * dt        -> heading theta
        x[k+1,3] = x[k,3] + u[k,1] * dt                 -> velocity u
    """

    def __init__(self, dt, process_sigma=[1.e-2,1.e-2,1.e-2,1.e-2]):
        self.process_sigma = process_sigma

        self.dt = dt
        self.nx = 4
        self.nu = 2
        self.ny = 4

        self.W = np.diag(process_sigma) ** 2

    def step(self, x, u):
        """
        Propagate state by one step and add noise
        """
        x_mean = x + np.array([
            x[3] * np.cos(x[2]) * self.dt,
            x[3] * np.sin(x[2]) * self.dt,
            u[0] * x[3] * self.dt,
            u[1] * self.dt
            ])
        return x_mean + np.random.multivariate_normal(np.zeros(self.nx), self.W)
        
    def mean_step(self, x, u):
        """
        Expected value of next state (no white noise is added)
        """
        x_mean = x + np.array([
            x[3] * np.cos(x[2]) * self.dt,
            x[3] * np.sin(x[2]) * self.dt,
            u[0] * x[3] * self.dt,
            u[1] * self.dt
            ])
        return x_mean 
    
    def linearize(self, x, u):
        A = np.eye(self.nx) + np.array([
            [0, 0, -x[3] * np.sin(x[2]) * self.dt, np.cos(x[2]) * self.dt],
            [0, 0,  x[3] * np.cos(x[2]) * self.dt, np.sin(x[2]) * self.dt],
            [0, 0,                              0,         u[0] * self.dt],
            [0, 0,                              0,                      0]
            ])
        B = np.array([[             0,       0],
                      [             0,       0],
                      [x[3] * self.dt,       0],
                      [             0, self.dt]])
        d = - A @ x - B @ u + self.mean_step(x, u)

        return A, B, d


    def linearize_around_trajectory(self, x, u):
        N = x.shape[0]
        if not x.shape[1] == self.nx:
            raise ValueError('trajectory x must have shape: (N,self.nx)')
        if not u.shape[1] == self.nu:
            raise ValueError('trajectory u must have shape: (N,self.nu)')

        A = np.zeros((N, self.nx, self.nx))
        B = np.zeros((N, self.nx, self.nu))
        d = np.zeros((N, self.nx))

        for k in range(N):
            A[k] = np.eye(self.nx) + np.array([
                [0, 0, -x[k,3] * np.sin(x[k,2]) * self.dt, np.cos(x[k,2]) * self.dt],
                [0, 0,  x[k,3] * np.cos(x[k,2]) * self.dt, np.sin(x[k,2]) * self.dt],
                [0, 0,                              0,         u[k,0] * self.dt],
                [0, 0,                              0,                      0]
                ])
            B[k] = np.array([[             0,       0],
                          [             0,       0],
                          [x[k,3] * self.dt,       0],
                          [             0, self.dt]])
            d[k] = - A[k] @ x[k] - B[k] @ u[k] + self.mean_step(x[k], u[k])

        return A, B, d

