
README — Projet d’Optimisation (Cônes)
1. Exécutables disponibles

Le projet contient les fichiers exécutables suivants :

Fichier	Description
probleme_cone.py	Définit le problème : f(r,h), contrainte c(r,h), Lagrangienne, gradients.
methode1.py	Méthode 1 : Descente de gradient à pas fixe.
methode2.py	Méthode 2 : Gradient + recherche linéaire (conditions de Wolfe).
methode3.py	Méthode 3 : Méthode de Newton appliquée à la Lagrangienne.
methode4.py	Méthode 4 : Quasi-Newton BFGS.
main.py	Programme principal permettant d’exécuter et comparer les 4 méthodes.

Pour lancer toutes les méthodes en une fois, exécuter uniquement :

python main.py (par terminal ou vscodium à Linux ordinateur)

2. Dépendances nécessaires

Le projet utilise uniquement NumPy.

Module	Version recommandée
python	≥ 3.8
numpy	≥ 1.20

Installation si nécessaire :

pip install numpy 


Aucune autre dépendance n’est requise.

3. Instructions d’exécution
Exécution du programme principal :
python main.py


Cela lance successivement :

 la méthode 1

 la méthode 2

 la méthode 3

 la méthode 4

et affiche pour chacune :

 les itérations

 le triplet final (r*, h*, λ*)

 la contrainte résiduelle c(r*,h*)

 la valeur f(r*,h*)

Exécution d’une méthode spécifique

Vous pouvez aussi exécuter une méthode individuellement :

 python methode1.py
 python methode2.py
 python methode3.py
 python methode4.py
