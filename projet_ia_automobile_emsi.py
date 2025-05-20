import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.arima.model import ARIMA

# ------------------------------- #
# Variables globales
donnees_generees = None

# ------------------------------- #
# Setup Main Window
root = tk.Tk()
root.title("Application IA Automobile - EMSI")
root.geometry("1000x700")
root.configure(bg="#0c2746")

# ------------------------------- #
# Load EMSI Logo Top Left
try:
    img = Image.open("C:/Users/Black/Pictures/logoemsi.png")
    img = img.resize((200, 100))
    photo = ImageTk.PhotoImage(img)

    frame_logo = tk.Frame(root, bg="#0c2746")
    frame_logo.place(x=10, y=10)

    label_logo = tk.Label(frame_logo, image=photo, bg="#0c2746")
    label_logo.image = photo
    label_logo.pack()
except Exception as e:
    print(f"Erreur de chargement de l'image : {e}")

# ------------------------------- #
# Title and Main Buttons
frame_main = tk.Frame(root, bg="#0c2746")
frame_main.pack(pady=30)

label_intro = tk.Label(
    frame_main,
    text="Application IA pour le secteur Automobile - EMSI",
    font=("Helvetica", 16, "bold"),
    fg="white",
    bg="#0c2746"
)
label_intro.pack(pady=20)

frame_buttons = tk.Frame(frame_main, bg="#0c2746")
frame_buttons.pack(pady=10)

# ------------------------------- #
# Algorithm Frame
frame_algo_buttons = tk.Frame(root, bg="#0c2746")
frame_plot = tk.Frame(root, bg="#0c2746")

fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=frame_plot)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Zone de texte pour l'explication des algorithmes
text_description = tk.Text(frame_plot, height=20, bg="#1e293b", fg="white", font=("Helvetica", 22), wrap="word")
text_description.pack(fill=tk.BOTH, expand=False, pady=10)

# ------------------------------- #
# Hover Effects for Buttons
def on_enter(e):
    e.widget['bg'] = '#1f6feb'

def on_leave(e):
    if e.widget['text'] == "Entrée":
        e.widget['bg'] = '#2563eb'
    else:
        e.widget['bg'] = '#dc2626'

# ------------------------------- #
# Display Algorithm and Graph Frame
def afficher_algorithmes():
    label_intro.pack_forget()
    frame_algo_buttons.pack(pady=10)
    frame_plot.pack(fill=tk.BOTH, expand=True)

# ------------------------------- #
# Plot Update Function
def update_plot(title, x_label, y_label):
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

# ------------------------------- #
# Fonction pour afficher une description
def afficher_description(texte):
    text_description.config(state=tk.NORMAL)
    text_description.delete("1.0", tk.END)
    text_description.insert(tk.END, texte)
    text_description.config(state=tk.DISABLED)

# ------------------------------- #
# Génération de Données (3 entrées, 1 sortie)
def generer_donnees():
    global donnees_generees

    try:
        nb_lignes = int(entry_nb.get())
        if nb_lignes <= 0:
            raise ValueError("Le nombre de lignes doit être positif.")
    except Exception as e:
        messagebox.showerror("Erreur", f"Valeur invalide : {e}")
        return

    temp = np.random.uniform(70, 120, nb_lignes)
    pression = np.random.uniform(1.0, 5.0, nb_lignes)
    vibration = np.random.uniform(10, 50, nb_lignes)

    consommation = 0.05 * temp + 0.2 * pression + 0.1 * vibration + np.random.normal(0, 2, nb_lignes)

    X = np.column_stack((temp, pression, vibration))
    y = consommation
    donnees_generees = (X, y)

    update_plot("Données Générées", "Observation", "Consommation")
    ax.plot(y, 'o', label='Consommation simulée')
    ax.legend()
    canvas.draw()

    afficher_description("📊 Données générées aléatoirement : température, pression, vibration comme entrées, et consommation estimée comme sortie. Ces données servent de base aux différents algorithmes d'IA.")

# ------------------------------- #
# Algorithms
def run_regression():
    global donnees_generees
    if donnees_generees:
        X, y = donnees_generees
    else:
        X = np.random.rand(100, 3)
        y = np.random.rand(100)

    model = LinearRegression().fit(X, y)
    pred_y = model.predict(X)

    update_plot("Régression Linéaire", "Observation", "Consommation")
    ax.plot(y, 'o', label='Réel')
    ax.plot(pred_y, 'x', label='Prédit')
    ax.legend()
    canvas.draw()

    afficher_description("🔍 La régression linéaire est un modèle prédictif qui permet d'estimer une valeur numérique (comme la consommation d'une voiture) à partir de plusieurs variables (température, pression, vibration).")

def run_clustering():
    global donnees_generees
    if donnees_generees:
        X, y = donnees_generees
        data = X
    else:
        data = np.random.rand(100, 3)

    model = KMeans(n_clusters=2).fit(data)
    labels = model.labels_

    update_plot("Clustering", "Observation", "Cluster")
    ax.scatter(range(len(labels)), labels, c=labels, cmap='Set2')
    canvas.draw()

    afficher_description("🧠 Le clustering regroupe les données similaires en différents groupes appelés clusters. Ici, l'algorithme KMeans divise les données en deux groupes selon leur ressemblance.")

def run_arima():
    global donnees_generees
    if donnees_generees:
        series = donnees_generees[1].tolist()
    else:
        series = np.random.rand(10).tolist()

    model = ARIMA(series, order=(1, 1, 0)).fit()
    forecast = model.forecast(steps=3)

    update_plot("Prévision ARIMA", "Temps", "Consommation")
    ax.plot(range(len(series)), series, marker='o', label='Historique')
    ax.plot(range(len(series), len(series) + len(forecast)), forecast, marker='x', linestyle='dashed', label='Prévision')
    ax.legend()
    canvas.draw()

    afficher_description("📈 Le modèle ARIMA est utilisé pour prévoir l’évolution de données dans le temps, comme la consommation. Il se base sur les données passées pour estimer les valeurs futures.")

def run_random_forest():
    global donnees_generees
    if donnees_generees:
        X = donnees_generees[0]
        y = (donnees_generees[1] > np.mean(donnees_generees[1])).astype(int)
    else:
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)

    model = RandomForestClassifier().fit(X, y)
    update_plot("Random Forest", "Classe", "Importance")
    ax.bar(["Classe 0", "Classe 1"], [0.5, 0.5], color=["green", "red"])
    canvas.draw()

    afficher_description("🌳 Random Forest est un algorithme d'apprentissage supervisé qui utilise plusieurs arbres de décision. Il permet ici de classifier les données en deux classes : faible ou forte consommation.")

def run_cross_validation():
    global donnees_generees
    if donnees_generees:
        X = donnees_generees[0]
        y = (donnees_generees[1] > np.mean(donnees_generees[1])).astype(int)
        model = DecisionTreeClassifier()
        scores = cross_val_score(model, X, y, cv=5)
    else:
        X, y = load_iris(return_X_y=True)
        model = DecisionTreeClassifier()
        scores = cross_val_score(model, X, y, cv=5)

    update_plot("Validation Croisée", "Score", "Fréquence")
    ax.hist(scores, bins=5, color='purple', edgecolor='black')
    canvas.draw()

    afficher_description("📊 La validation croisée évalue la performance d'un modèle d’IA en testant sa précision sur plusieurs sous-ensembles de données. Cela permet de vérifier sa robustesse.")

# ------------------------------- #
# Styled Buttons
button_style = {
    "width": 20,
    "height": 1,
    "relief": "flat",
    "font": ("Helvetica", 11, "bold"),
    "bd": 0,
    "activeforeground": "white",
    "cursor": "hand2"
}

btn_entree = tk.Button(
    frame_buttons,
    text="Entrée",
    bg="#2563eb",
    fg="white",
    command=afficher_algorithmes,
    **button_style
)
btn_sortie = tk.Button(
    frame_buttons,
    text="Sortie",
    bg="#dc2626",
    fg="white",
    command=root.destroy,
    **button_style
)

btn_entree.bind("<Enter>", on_enter)
btn_entree.bind("<Leave>", on_leave)
btn_sortie.bind("<Enter>", on_enter)
btn_sortie.bind("<Leave>", on_leave)

btn_entree.pack(side=tk.LEFT, padx=20, pady=10)
btn_sortie.pack(side=tk.RIGHT, padx=20, pady=10)

# ------------------------------- #
# Entry for number of lines
entry_frame = tk.Frame(frame_algo_buttons, bg="#0c2746")
entry_frame.grid(row=0, column=0, columnspan=2, pady=10)

tk.Label(entry_frame, text="Nombre de lignes :", bg="#0c2746", fg="white").pack(side=tk.LEFT, padx=5)
entry_nb = tk.Entry(entry_frame, width=5)
entry_nb.insert(0, "200")
entry_nb.pack(side=tk.LEFT)

# ------------------------------- #
# Algorithm Buttons
algo_buttons = [
    ("Générer des Données", generer_donnees),
    ("Régression Linéaire", run_regression),
    ("Clustering", run_clustering),
    ("ARIMA", run_arima),
    ("Random Forest", run_random_forest),
    ("Validation Croisée", run_cross_validation)
]

for i, (name, func) in enumerate(algo_buttons):
    tk.Button(
        frame_algo_buttons,
        text=name,
        command=func,
        **button_style,
        bg="#10b981" if name == "Générer des Données" else "yellow",
        fg="white" if name == "Générer des Données" else "black"
    ).grid(row=1 + i // 2, column=i % 2, padx=11, pady=10)

# ------------------------------- #
root.mainloop()
