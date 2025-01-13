# Projet de Statistiques & Probabilités

Ce projet est une application web développée avec Django qui permet de visualiser et d'analyser des données statistiques à partir de fichiers Excel, CSV . Il propose également des outils pour effectuer des tests statistiques et des calculs de distributions.

## Prérequis

- Python 3.x
- pip (gestionnaire de paquets Python)

## Installation

1. Clonez le dépôt du projet :

    ```bash
    git clone <URL_DU_DEPOT>
    cd <NOM_DU_REPERTOIRE>
    ```

2. Créez un environnement virtuel et activez-le :

    ```bash
    python -m venv env
    source env/bin/activate  # Sur Windows: env\Scripts\activate
    ```

3. Installez les dépendances requises :

    ```bash
    pip install -r requirements.txt
    ```

4. Appliquez les migrations de la base de données :

    ```bash
    python manage.py migrate
    ```

5. Lancez le serveur de développement :

    ```bash
    python manage.py runserver
    ```

6. Ouvrez votre navigateur et accédez à l'adresse suivante :

    ```
    http://127.0.0.1:8000/
    ```

## Fonctionnalités

- **Importation de fichiers** : Importez des fichiers Excel, CSV pour visualiser et analyser les données.
- **Visualisation de données** : Créez différents types de graphiques (barplot, histogramme, pie chart, scatter plot, heatmap, etc.) à partir des données importées.
- **Filtrage de données** : Filtrez les données en fonction des colonnes et des valeurs.
- **Calculs statistiques** : Calculez des Descripteurs statistiques (moyenne, médiane, variance, écart-type, etc.).
- **Tests statistiques** : Effectuez des tests statistiques tels que le test t, le test z, et la régression linéaire.
- **Calculs de distributions** : Calculez et visualisez des distributions statistiques (binomiale, Bernoulli, normale, Poisson, uniforme, exponentielle).

## Structure du Projet

- `core/` : Contient les vues, les formulaires et les URL de l'application.
- `templates/` : Contient les fichiers HTML pour le rendu des pages.
- `static/` : Contient les fichiers CSS et JavaScript.
- `requirements.txt` : Liste des dépendances Python nécessaires pour le projet.

## Auteurs

- **Ketaj Youssef**
- **Farid Oussama**
