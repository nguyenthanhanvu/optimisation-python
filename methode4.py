import numpy as np
from probleme_cone import f_val, c_val, grad_f_rh, grad_c_rh, L_val, grad_L_rh

# Recherche linéaire de Wolfe sur la Lagrangienne L(r,h,λ)


def wolfe_step(r: float, h: float, lam: float,
               d: np.ndarray,
               c1: float = 1e-4,
               c2: float = 0.9,
               a0: float = 1.0) -> float:
    """
    Recherche de pas satisfaisant les conditions de Wolfe :
        1) Armijo  : L(x+αd) ≤ L(x) + c1 α ∇L(x)^T d
        2) Courbure: ∇L(x+αd)^T d ≥ c2 ∇L(x)^T d
    """
    L0 = L_val(r, h, lam)
    g0 = grad_L_rh(r, h, lam)
    dphi0 = np.dot(g0, d)   # dérivée directionnelle au point de départ

    # si la direction n'est pas de descente, on renvoie α=0
    if dphi0 >= 0:
        return 0.0

    # bornes initiales et pas initial
    amin = 0.0
    amax = 100.0
    a = a0

    cond1 = 0
    cond2 = 0
    it = 0
    it_max = 50

    while (cond1 + cond2) < 2 and it < it_max:
        it += 1
        r_new = r + a * d[0]
        h_new = h + a * d[1]

        # rester dans le domaine r>0, h>0
        if r_new <= 0 or h_new <= 0:
            amax = a
            a = 0.5 * (amin + amax)
            cond1 = 0
            cond2 = 0
            continue

        L_a = L_val(r_new, h_new, lam)
        g_a = grad_L_rh(r_new, h_new, lam)
        dphia = np.dot(g_a, d)

        # --- Armijo ---
        if L_a <= L0 + c1 * a * dphi0:
            cond1 = 1
        else:
            cond1 = 0
            # trop grand → on diminue a
            amax = a
            a = 0.5 * (amin + amax)
            cond2 = 0
            continue

        # --- condition de courbure (Wolfe) ---
        if dphia >= c2 * dphi0:
            cond2 = 1
        else:
            cond2 = 0
            # pente encore trop négative → on augmente a
            amin = a
            a = 2.0 * a
            continue

    return a


# Méthode 4 : quasi-Newton BFGS sur (r,h), avec mise à jour de λ

def methode4_bfgs(r0: float = 1.0,
                  h0: float = 1.5,
                  lam0: float = 25.0,
                  tol: float = 1e-6,) -> np.ndarray:
    """
    Méthode de quasi-Newton BFGS appliquée à la Lagrangienne L(r,h,λ).

    Variables primales : x = (r,h).
    On approxime H_L^{-1}(x,λ) par B.
    À chaque itération :
        - direction d_k = -B_k ∇_x L(x_k, λ_k)
        - recherche linéaire de Wolfe pour α_k
        - x_{k+1} = x_k + α_k d_k
        - mise à jour B (formule BFGS) avec s = x_{k+1}-x_k, y = g_{k+1}-g_k
        - mise à jour du multiplicateur : λ_{k+1} = λ_k - c(r_{k+1},h_{k+1})"""

    # initialisation
    x = np.array([float(r0), float(h0)])
    lam = float(lam0)

    g = grad_L_rh(x[0], x[1], lam)     # gradient initial
    B = np.eye(2)                      # approximation initiale de H^{-1}
    k = 0

    while True:
        c_k = c_val(x[0], x[1])
        norm_g = np.linalg.norm(g)

        print(
            f"it {k:03d} | r={x[0]:.9f}, h={x[1]:.9f}, λ={lam:.9f} "
            f"| c={c_k:+.3e} | ||∇L||={norm_g:.3e}")

        # critère d'arrêt : stationnarité + contrainte ≃ 0
        if abs(c_k) <= tol and norm_g <= tol:
            print(f"Convergence atteinte à l'itération {k}")
            break

        # 1) direction de recherche BFGS : d = -B g

        d = -B.dot(g)

        # si ce n'est pas une direction de descente, on "réinitialise"
        if np.dot(d, g) >= 0:
            # direction fallback : gradient négatif
            d = -g
            B = np.eye(2)

        # 2) recherche linéaire de Wolfe pour α
        alpha = wolfe_step(x[0], x[1], lam, d)

        # 3) mise à jour de x = (r,h)
        s = alpha * d
        x_new = x + s

        # r,h > 0
        x_new[0] = max(x_new[0], 1e-12)
        x_new[1] = max(x_new[1], 1e-12)

        # gradient au nouveau point, λ encore inchangé
        g_new = grad_L_rh(x_new[0], x_new[1], lam)

        # 4) Mise à jour de B (formule BFGS classique pour H^{-1}) :
#    s = x_{k+1} - x_k : déplacement en x
#    y = g_{k+1} - g_k : variation du gradient
#    On vérifie la condition de courbure y^T s > 0 pour conserver B_k
#    symétrique définie positive.
        y = g_new - g
        ys = float(np.dot(y, s))

        if ys > 0:
            # ρ = 1 / (y^T s)
            rho = 1.0 / ys
            I = np.eye(2)  # Matrice unitaire
    # B_{k+1} = (I - ρ s y^T) B_k (I - ρ y s^T) + ρ s s^T
            B = (I - rho * np.outer(s, y)) @ B @ (I - rho * np.outer(y, s)) \
                + rho * np.outer(s, s)

        else:
            print("  (mise à jour BFGS ignorée : y^T s <= 0)")

        # 5) mise à jour de λ à partir de la contrainte

        c_k_plus = c_val(x_new[0], x_new[1])
        lam = lam - c_k_plus

        # next
        x = x_new
        g = grad_L_rh(x[0], x[1], lam)   # gradient avec nouveau λ
        k += 1

    return np.array([x[0], x[1], lam])


# petit test si on lance directement ce fichier
if __name__ == "__main__":
    sol = methode4_bfgs()
    print("\nSolution Méthode 4 (BFGS quasi-Newton) :", sol)
