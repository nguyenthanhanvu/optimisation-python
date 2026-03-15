import numpy as np

# ------- Données du problème -------
V0 = 1.0  # volume imposé


def f_val(r, h):
    """Surface latérale au carré."""
    return (np.pi**2)*(r**2)*(r**2 + h**2)


def c_val(r, h):
    """Contraintes de volume : c(r,h) = 0 à l’optimum."""
    return (np.pi/3.0)*r**2*h - V0


def grad_f_rh(r, h):
    """Gradient de f par rapport à (r, h)."""
    df_dr = 4*np.pi**2*r**3 + 2*np.pi**2*h**2*r
    df_dh = 2*np.pi**2*h*r**2
    return np.array([df_dr, df_dh])


def grad_c_rh(r, h):
    """Gradient de c par rapport à (r, h)."""
    dc_dr = (2*np.pi/3.0)*r*h
    dc_dh = np.pi/3.0*r**2
    return np.array([dc_dr, dc_dh])


def L_val(r: float, h: float, lam: float) -> float:
    """ Calcule la valeur de la Lagrangienne L(r,h,λ) = f(r,h) - λ c(r,h). """
    return f_val(r, h) - lam * c_val(r, h)


def grad_L_rh(r: float, h: float, lam: float) -> np.ndarray:
    """Gradient de la Lagrangienne par rapport aux variables primales (r, h) """
    return grad_f_rh(r, h) - lam * grad_c_rh(r, h)
