import numpy as np
from probleme_cone import f_val, c_val, grad_f_rh, grad_c_rh, L_val, grad_L_rh

# On travaille ici avec la Lagrangienne :
#     L(r, h, λ) = f(r, h) - λ c(r, h)


#  Recherche linéaire de Wolfe (Cond1 + Cond2 avec cond1 / cond2 explicites)
""" Conditions de Wolfe:
      1) Armijo  : L(x+ad) ≤ L(x) + c1 α L'(0)
      2) Courbure: L'(a) ≥ c2 L'(0)"""


def wolfe_step(r: float, h: float, lam: float, d: np.ndarray, c1: float = 1e-4,
               c2: float = 0.9, a0: float = 1e-3) -> float:
    L0 = L_val(r, h, lam)  # L(0)
    g0 = grad_L_rh(r, h, lam)  # ∇L(0)
    dphi0 = np.dot(g0, d)   # ψ'(0)

    # Bornes initiales
    amin = 0.0        # a_min = 0
    amax = 100.0      # a_max = 100
    a = a0            # a_k = 1e-3

    cond1 = 0
    cond2 = 0
    it = 0
    it_max = 50  # <- Limite du nombre d'itérations pour éviter boucle infinie

    while (cond1 + cond2) < 2 and it < it_max:  # Tant que les 2 conditions ne sont pas satisfaites
        it += 1  # <- compteur d'itérations
        r_new = r + a * d[0]  # Nouveau r
        h_new = h + a * d[1]  # Nouveau h

        if r_new <= 0 or h_new <= 0:  # On reste dans le domaine défini par r>0 et h>0
            amax = a  # On réduit la borne supérieure
            a = 0.5 * (amin + amax)  # On prend le milieu des bornes
            cond1 = 0  # On réinitialise les conditions
            cond2 = 0  # On réinitialise les conditions
            continue

        L_a = L_val(r_new, h_new, lam)  # L(a)
        g_a = grad_L_rh(r_new, h_new, lam)  # ∇L(a)
        dphi_a = np.dot(g_a, d)  # ψ'(a)

        # ---- Cond1 : Armijo ----
        if L_a <= L0 + dphi0 * a * c1:  # Si la condition d'Armijo est satisfaite
            cond1 = 1  # On valide la condition 1
        else:
            cond1 = 0  # On invalide la condition 1
            amax = a  # On réduit la borne supérieure
            a = 0.5 * (amin + amax)  # On prend le milieu des bornes
            continue

        # ---- Cond2 : courbure (Wolfe) ----
        if dphi_a >= dphi0 * c2:  # Si la condition de courbure est satisfaite
            cond2 = 1  # On valide la condition 2
        else:
            cond2 = 0  # On invalide la condition 2
            amin = a  # On augmente la borne inférieure
            a = 2.0 * a  # On double le pas
            continue
    return a

# -------- METHOD 2: Gradient + Wolfe -------- #


def method2(
    r0=1.0, h0=1.5, lam0=25.0,
    tol=1e-6  # Tolérance de convergence
):

    r, h, lam = r0, h0, lam0
    k = 0
    c = c_val(r, h)  # contrainte de volume
    lam = lam - c
    # λ doit s'ajuster pour imposer progressivement la contrainte c(r,h)=0.
    # Si c > 0 : le volume est trop grand  → λ diminue.
    # Si c < 0: le volume est trop petit → λ augmente.
    # La mise à jour λ ← λ - c correspond à une étape de correction

    # ----- Check convergence -----
    stat = np.linalg.norm(grad_f_rh(r, h) - lam*grad_c_rh(r, h))
    while abs(c) > tol or stat > tol:  # convergence atteinte si il ne rêpète plus
        # ----- Gradient minimization of L -----
        g = grad_L_rh(r, h, lam)

        d = -g
        a = wolfe_step(r, h, lam, d)

        r = r + a*d[0]  # pour eviter r,h = 0
        h = h + a*d[1]  # pour eviter r,h = 0

        # On impose r,h > 0 pour garder un cône géométriquement valide
        r = max(r, 1e-12)
        h = max(h, 1e-12)

        # ----- Update λ (dual correction step) -----
        # correction simple pour faire tendre la contrainte vers 0
        c = c_val(r, h)
        lam = lam - c

        # ----- Check convergence -----
        stat = np.linalg.norm(grad_f_rh(r, h) - lam*grad_c_rh(r, h))
        k += 1
        print(
            f"it {k:02d} | r={r:.9f}, h={h:.9f}, λ={lam:.9f} | c={c:+.3e} | station={stat:.3e}")
    print(f"converge atteinte à l'itération {k}")
    return np.array([r, h, lam])


if __name__ == "__main__":
    sol = method2()
    print("\nFinal (Method 2 – Armijo + Wolfe):", sol)
