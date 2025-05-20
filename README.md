# 🧠 Application Desktop d'Intelligence Artificielle – Secteur Automobile

Projet réalisé dans le cadre du module **Modèles Statistiques de l’Intelligence Artificielle** – EMSI (3ème année Génie Informatique).  
Encadrante : **Dr. El Mkhalet Mouna**  
Réalisé par : **Meryem Mar**

---

## 🎯 Objectifs du Projet

- Développer une application **desktop intelligente et pédagogique** avec interface graphique (`Tkinter`)
- Simuler des **données industrielles automobiles**
- Intégrer plusieurs **algorithmes de machine learning**
- Afficher les **résultats de manière interactive et visuelle**
- Fournir des **explications accessibles** pour chaque algorithme
- Exporter l'application au format `.exe` et la **publier sur GitHub**

---

## 🖥️ Interface Utilisateur

L'application comprend :

- 🎨 Une interface claire, ergonomique, et colorée
- 📊 Des boutons pour chaque algorithme
- 📈 Une zone graphique intégrée (`matplotlib`)
- 🧾 Une zone de texte dynamique qui explique le fonctionnement de chaque modèle

![Aperçu de l'application](images/screenshot.png)

---

## 🧪 Algorithmes intégrés

| Algorithme           | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| Régression Linéaire  | Prédiction de la consommation à partir des capteurs (température, pression, vibration) |
| Clustering (KMeans)  | Regroupement non supervisé des données                                      |
| ARIMA                | Modèle de prévision des séries temporelles                                 |
| Random Forest        | Classification binaire (consommation haute/basse)                          |
| Validation Croisée   | Évaluation de la robustesse des modèles                                    |

---

## ⚙️ Technologies utilisées

- `Python 3`
- `Tkinter` — interface graphique
- `Scikit-learn` — algorithmes ML
- `Matplotlib` — visualisation
- `Statsmodels` — ARIMA
- `PyInstaller` — création de l’exécutable

---

## 🚀 Lancer l’application

### ▶️ En local (à partir du code source) :
```bash
pip install -r requirements.txt
python main.py
