import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import cholesky


def solve_differential_equations(init_cond, t_span):
    def model(y, t):
        x, v_x, z, v_z, theta = y
        dydt = [v_x,
                0.4 * abs(np.sin(t)) * np.cos(theta) + 0.2 * abs(np.cos(t)) * np.sin(theta),
                v_z,
                0.2 * abs(np.cos(t)) * np.cos(theta) - 0.4 * abs(np.sin(t)) * np.sin(theta),
                theta_dot_0]
        return dydt

    sol = odeint(model, init_cond, t_span)
    return sol.T  # Transpose to match the MATLAB structure.


def define_gps_vector(X_real, R):
    gps = X_real[:2, :] + np.sqrt(R) @ np.random.randn(2, X_real.shape[1])
    return gps


def perform_unscented_kalman_filter(dt, H, R, gps, t_span, P, X_real, init_cond):
    n = 5
    alpha = 1e-3
    beta = 2
    kappa = 3 - n
    lambda_ = alpha ** 2 * (n + kappa) - n

    X = np.zeros((n, len(t_span)))
    X[:, 0] = init_cond
    std_k = np.zeros_like(X)
    std_k[:, 0] = np.sqrt(np.diag(P))

    Wm = np.hstack([lambda_ / (n + lambda_), 0.5 / (n + lambda_) * np.ones(2 * n)])
    Wc = np.copy(Wm)
    Wc[0] += (1 - alpha ** 2 + beta)

    for i in range(len(t_span) - 1):
        sqrtP = cholesky((n + lambda_) * P).T
        sigma_pts = np.hstack([X[:, [i]], X[:, [i]] + sqrtP, X[:, [i]] - sqrtP])

        # Time Update (Prediction)
        X_pred = np.zeros(n)
        for j in range(2 * n + 1):
            sp = sigma_pts[:, j]
            sp_transformed = sp + dt * np.array([sp[2],
                                                 0.4 * abs(np.sin(t_span[i])) * np.cos(sp[4]) + 0.2 * abs(
                                                     np.cos(t_span[i])) * np.sin(sp[4]),
                                                 sp[3],
                                                 0.2 * abs(np.cos(t_span[i])) * np.cos(sp[4]) - 0.4 * abs(
                                                     np.sin(t_span[i])) * np.sin(sp[4]),
                                                 theta_dot_0])
            X_pred += Wm[j] * sp_transformed

        P_pred = np.zeros((n, n))
        for j in range(2 * n + 1):
            sp = sigma_pts[:, j]
            sp_transformed = sp + dt * np.array([sp[2],
                                                 0.4 * abs(np.sin(t_span[i])) * np.cos(sp[4]) + 0.2 * abs(
                                                     np.cos(t_span[i])) * np.sin(sp[4]),
                                                 sp[3],
                                                 0.2 * abs(np.cos(t_span[i])) * np.cos(sp[4]) - 0.4 * abs(
                                                     np.sin(t_span[i])) * np.sin(sp[4]),
                                                 theta_dot_0])
            P_pred += Wc[j] * np.outer(sp_transformed - X_pred, sp_transformed - X_pred)

        # Measurement Update
        z_pred = H @ X_pred
        residual_m = gps[:, i + 1] - z_pred

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        X[:, i + 1] = X_pred + K @ residual_m
        P = (np.eye(n) - K @ H) @ P_pred
        std_k[:, i + 1] = np.sqrt(np.diag(P))

    return X, std_k
def plot_results(gps, X, X_real):
    plt.plot(gps[0, :], gps[1, :], 'o',markerfacecolor='none', markeredgecolor='blue',  label='GPS Location')
    plt.plot(X[0, :], X[1, :], '*',markerfacecolor='none', markeredgecolor='orange', label='EKF Prediction')
    plt.plot(X_real[0, :], X_real[1, :], '-k', linewidth=0.8, label='Trajectory')
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.legend(loc='best')
    plt.show()

# Initialize Parameters and Initial Conditions
x_0 = 100
z_0 = 200
v_x_0 = 2
v_z_0 = 1
theta_0 = 0.01
theta_dot_0 = 0.01
init_cond = [x_0, v_x_0, z_0, v_z_0, theta_0]

P = np.diag([10, 15, 2, 3, 0.08]) ** 2
R = np.diag([8, 4]) ** 2
H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])

dt = 0.01
T = 40
t_span = np.arange(0, T + dt, dt)

# Solve Differential Equations
X_real = solve_differential_equations(init_cond, t_span)

# Define GPS Vector
gps = define_gps_vector(X_real, R)

# Perform Unscented Kalman Filter
X, std_k = perform_unscented_kalman_filter(dt, H, R, gps, t_span, P, X_real, init_cond)

# Plot The Result
plot_results(gps, X, X_real)
