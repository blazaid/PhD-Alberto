import numpy as np

def calibration_matrix(pitch, roll, yaw):
    alpha = roll * np.pi / 180.
    phi = pitch * np.pi / 180.
    theta = yaw * np.pi / 180.

    sin_alpha, cos_alpha = np.sin(alpha), np.cos(alpha)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)

    a = cos_phi * cos_theta - sin_phi * sin_alpha * sin_theta
    b = sin_theta * cos_phi + cos_theta * sin_alpha * sin_phi
    c = -sin_phi * cos_alpha
    d = -sin_theta * cos_alpha
    e = cos_alpha * cos_theta
    f = sin_alpha
    g = cos_theta * sin_phi + sin_theta * sin_alpha * cos_phi
    h = sin_theta * sin_phi - cos_theta * sin_alpha * cos_phi
    i = cos_alpha * cos_phi

    return np.array([
        [a, b, c, 0],
        [d, e, f, 0],
        [g, h, i, 0],
        [
            0 * a + 0 * d + 1.75 * g,
            0 * b + 0 * e + 1.75 * h,
            0 * c + 0 * f + 1.75 * i,
            1
        ],
    ])