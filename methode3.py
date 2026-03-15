import numpy as np
from probleme_cone import f_val, c_val, grad_f_rh, grad_c_rh, grad_L_rh

# Lagrangienne : L(r,h,λ) = f(r,h) - λ c(r,h)

# Hessienne de L : H_L = H_f - λ H_c


def hess_f_rh(r: float, h: float) -> np.ndarray:
    """Hessienne de f(r,h) = π² r² (r² + h²)

    H_f = [ d²f/dr²      d²f/dr dh ]
          [ d²f/dh dr    d²f/dh²  ]

    d²f/dr²   = 12 π² r² + 2 π² h²
    d²f/dr dh = 4  π² r h
    d²f/dh²   = 2  π² r² """
    pi2 = np.pi**2
    d2f_dr2 = 12*pi2*r**2 + 2*pi2*h**2
    d2f_drdh = 4*pi2*r*h
    d2f_dh2 = 2*pi2*r**2
    return np.array([[d2f_dr2,  d2f_drdh],
                     [d2f_drdh, d2f_dh2]])


def hess_c_rh(r: float, h: float) -> np.ndarray:
    """Hessienne de c(r,h) = (π/3) r² h - 1

    d²c/dr²   = (2π/3) h
    d²c/dr dh = (2π/3) r
    d²c/dh²   = 0 """

    d2c_dr2 = (2*np.pi/3.0)*h
    d2c_drdh = (2*np.pi/3.0)*r
    d2c_dh2 = 0
    return np.array([[d2c_dr2,  d2c_drdh],
                     [d2c_drdh, d2c_dh2]])


def hess_L_rh(r: float, h: float, lam: float) -> np.array:
    """Hessienne de la Lagrangienne :
        H_L = H_f - λ H_c """
    return hess_f_rh(r, h) - lam*hess_c_rh(r, h)

# Méthode 3 : Newton sur L(r,h,λ) avec pas unitaire (α_k = 1)


def methode3_newton(r0: float = 1.0,
                    h0: float = 1.5,
                    lam0: float = 25.0,
                    tol: float = 1e-6) -> np.ndarray:
    """
    Méthode 3 (Newton) :

    À chaque itération k :
        1) On calcule g_k = ∇_{r,h} L(r_k,h_k,λ_k)
        2) On calcule H_k = Hessienne de L
        3) Direction de Newton : d_k = - H_k^{-1} g_k
        4) Pas unitaire : (r_{k+1}, h_{k+1}) = (r_k, h_k) + d_k
        5) Mise à jour de λ :
               λ_{k+1} = λ_k - c(r_k, h_k)
           (correction simple pour faire tendre la contrainte vers 0)

    Remarque : aucune recherche de pas (α_k = 1),(ce rôle sera pris par la méthode 4, BFGS).
    """

    r, h, lam = float(r0), float(h0), float(lam0)
    k = 0
    # Gradient de L au point courant
    gL = grad_L_rh(r, h, lam)

    # norme du gradient pour critère de convergence
    norm_g = np.linalg.norm(gL)

    c = c_val(r, h)

    while norm_g > tol or abs(c_val(r, h)) > tol:  # tant que pas convergé

        HL = hess_L_rh(r, h, lam)  # matrice 2x2

        norm_g = np.linalg.norm(gL)

        # Direction de Newton : H_L d = -g  →  d = - H_L^{-1} g
        d = -np.linalg.solve(HL, gL)  # Résolution du système linéaire

        r = r + d[0]
        h = h + d[1]

        # On impose r,h > 0 pour garder un cône géométriquement valide
        r = max(r, 1e-12)
        h = max(h, 1e-12)

        gL = grad_L_rh(r, h, lam)  # Mise à jour du gL
        norm_g = np.linalg.norm(gL)  # Mise à jour du norm g

        # Mise à jour du multiplicateur de Lagrange
        c = c_val(r, h)
        lam = lam - c

        k += 1
        print(
            f"it {k:03d} | r={r:.9f}, h={h:.9f}, λ={lam:.9f} "
            f"| c={c:+.3e} | ||∇L||={norm_g:.3e}"
        )

    print(f"converge atteinte à l'itération {k}")

    return np.array([r, h, lam])


# Test rapide si on lance directement ce fichier
if __name__ == "__main__":
    sol = methode3_newton()
    print("\nSolution Méthode 3 (Newton) :", sol)
