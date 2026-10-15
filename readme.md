# Live coding Tensorflow keras
L'objectif est de construire ton premier réseau de neurones pour un problème de classification, de l'entraîner, puis d'interpréter ses performances comme dans un vrai projet de Data Science.

**dataset**
on va utiliser un dataset de détection de tumeurs cérébrales (Brain Tumor MRI).

The dataset contains 7,023 human brain MRI images categorized into four classes:

Glioma
Meningioma
Pituitary tumor
No tumor
The dataset is structured into four folders for each category.

```text
L’objectif sera de construire un réseau de neurones qui classe des images IRM du cerveau en deux catégories : avec tumeur et sans tumeur, selon les classes réellement disponibles dans le dataset choisi.
```

**installation des requrements**
```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## étapes
1. **Charger et explorer les images:**
On utilisera image_dataset_from_directory, qui charge les images depuis le dossier data et attribue automatiquement les labels à partir des noms de classes.
puis afficher quelques images pour vérifier les données.

**Objectifs :**
- Charger et explorer le dataset.
- Identifier les entrées X (images) et la cible y (classes).
- Séparer les données en entraînement et en test.
- Préparer les images pour l'apprentissage.

2. **Construction du modèle:**
On construit un CNN simple avec des couches Dense pour la classification finale, conformément à l'objectif de créer un premier réseau de neurones.

**Objectifs :**
- Construction d'un réseau (CNN) avec des couches Con2D et Dense.
- Choix des fonctions d’activation adaptées (dans notre cas ReLU, SoftMax pour la couche de sortie).
- Définition de la couche de sortie avec 4 sorties.
- Compilation de modèle avec un optimiseur(adam) et une fonction de perte(sparse_categorical_crossentropy).

**Architecture :**
- Conv2D extrait des caractéristiques visuelles des images.
- MaxPooling2D réduit la taille des représentations.
- Flatten transforme les caractéristiques en vecteur.
- Dense apprend à associer ces caractéristiques aux classes.
- softmax produit une probabilité pour chaque classe.

3. **Entraînement:**
- epochs=10 : nombre maximal de passages sur les données d'entraînement.
- batch_size=32 : nombre d'images traitées par lot. Ici, la taille des lots est déjà définie lors du chargement du dataset.

4. **Évaluation:**
Cette partie passe par:
- Évaluation du modèle sur les données de test.
- Visualisation des courbes d’apprentissage.
- Identification d'un éventuel overfitting ou underfitting si existant.
- Réalisation d'une prédiction sur une nouvelle donnée.