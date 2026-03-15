import numpy as np

from methode1 import method1
from methode2 import method2
from methode3 import methode3_newton
from methode4 import methode4_bfgs
from probleme_cone import f_val, c_val  # pour évaluer f* et la contrainte


def afficher_resultat(nom_methode, sol):
    """Affiche proprement r*, h*, λ*, f* et la contrainte résiduelle."""
    r, h, lam = sol
    f_star = f_val(r, h)
    c_res = c_val(r, h)

    print("\n" + "="*70)
    print(f"{nom_methode}")
    print("-"*70)
    print(f"r* = {r:.9f}")
    print(f"h* = {h:.9f}")
    print(f"λ* = {lam:.9f}")
    print(f"f(r*,h*)= {f_star:.9e}")
    print(f"c(r*,h*)= {c_res:+.3e} (doit être ≈ 0)")
    print("="*70)


def main():
    # Même conditions initiales pour toutes les méthodes
    r0 = 1.0
    h0 = 1.5
    lam0 = 25.0

    print(" PROGRAMME PRINCIPAL – COMPARAISON DES 4 MÉTHODES ")

    # Méthode 1 : gradient à pas fixe
    print(" Lancement de la Méthode 1 (gradient à pas fixe)")
    sol1 = method1(r0=r0, h0=h0, lam0=lam0)
    afficher_resultat("Méthode 1 – Gradient à pas fixe", sol1)

    # Méthode 2 : gradient + Wolfe
    print(" Lancement de la Méthode 2 (gradient + Wolfe)")
    sol2 = method2(r0=r0, h0=h0, lam0=lam0)
    afficher_resultat("Méthode 2 – Gradient + Wolfe", sol2)

    # Méthode 3 : Newton
    print(" Lancement de la Méthode 3 (Newton)")
    sol3 = methode3_newton(r0=r0, h0=h0, lam0=lam0)
    afficher_resultat("Méthode 3 – Newton sur L", sol3)

    # Méthode 4 : quasi-Newton BFGS
    print(" Lancement de la Méthode 4 (BFGS)")
    sol4 = methode4_bfgs(r0=r0, h0=h0, lam0=lam0)
    afficher_resultat("Méthode 4 – quasi-Newton BFGS", sol4)

    print(" FIN DU PROGRAMME PRINCIPAL ")


if __name__ == "__main__":
    main()
