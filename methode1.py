import numpy as np
from probleme_cone import (
    f_val,  # fonction objectif S(r,h)
    c_val,   # contrainte de volume c(r,h)
    grad_f_rh,  # ∇f(r,h)
    grad_c_rh,  # ∇c(r,h)
    L_val,  # L(r,h,λ)
    grad_L_rh  # ∇_r,h L(r,h,λ))
)

#  MÉTHODE 1 : MÉTHODE DE GRADIENT À PAS FIXE
# Schéma
#  Tant que ||∇L(r,h,λ)|| ≥ ε :
#  1) d_k = -∇_r,h L(r_k, h_k, λ_k)
#  2) r_{k+1} = r_k + α d_k(0),  h_{k+1} = h_k + α d_k(1)  (α fixe)
#  3) Mise à jour de λ pour faire tendre c(r,h) vers 0


def method1(
    r0: float = 1.0,
    h0: float = 1.5,
    lam0: float = 25.0,
    alpha: float = 1e-3,   # pas fixe α
    tol: float = 1e-6,     # tolérance de convergence
) -> np.ndarray:

    # Initialisation
    r, h, lam = r0, h0, lam0

    # première correction duale simple
    c = c_val(r, h)
    lam = lam - c

    # critères de convergence (contrainte & stationnarité)
    stat = np.linalg.norm(grad_f_rh(r, h) - lam * grad_c_rh(r, h))
    k = 0
    while (abs(c) > tol or stat > tol):

        # 1) Direction de descente : gradient de L par rapport à (r,h)
        g = grad_L_rh(r, h, lam)
        d = -g   # descente de gradient

    # 2) Mise à jour à pas fixe
        r = r + alpha * d[0]
        h = h + alpha * d[1]

    # On impose r,h > 0 (cône géométriquement valide)
        r = max(r, 1e-12)
        h = max(h, 1e-12)

    # 3) Mise à jour du multiplicateur λ (correction duale)
        c = c_val(r, h)
        lam = lam - c

    # 4) Mise à jour des critères de convergence
        stat = np.linalg.norm(grad_f_rh(r, h) - lam * grad_c_rh(r, h))
        k += 1

        print(
            f"it {k:04d} | r={r:.9f}, h={h:.9f}, λ={lam:.9f} | "
            f"c={c:+.3e} | station={stat:.3e}"
        )

    print(f" Convergence atteinte à l'itération {k} (méthode 1 – pas fixe).")

    return np.array([r, h, lam])


if __name__ == "__main__":
    sol1 = method1()
    print("\nFinal (Méthode 1 – gradient à pas fixe):", sol1)
