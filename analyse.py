import pandas as pd
import matplotlib.pyplot as plt

# 1. Chargement
df = pd.read_csv("IRIS.csv")

# 2. Colonnes attendues
target_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# 3. Nettoyage
# Conversion en numérique
for col in target_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Suppression doublons
nb_doublons = df.duplicated().sum()
if nb_doublons > 0:
    df.drop_duplicates(inplace=True)
    print(f"{nb_doublons} doublons supprimés.")

# Suppression lignes avec valeurs manquantes dans les 4 colonnes
lignes_avant = len(df)
df.dropna(subset=target_cols, inplace=True)
lignes_apres = len(df)

if lignes_avant - lignes_apres > 0:
    print(f"{lignes_avant - lignes_apres} lignes incomplètes supprimées.")

# 4. Export
df.to_csv("iris_clean.csv", index=False)
print("Succès ! Fichier généré : iris_clean.csv")
print(f"Taille finale : {df.shape[0]} lignes.\n")

# Variables graphiques
cols = target_cols
titres = ["Longueur Sépale", "Largeur Sépale", "Longueur Pétale", "Largeur Pétale"]

# FIGURE 1: Moyennes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    moy = df.groupby("species")[cols[i]].mean()
    moy.plot(kind="bar", ax=ax, color="#9b5de5")
    ax.set_title(titres[i])
    ax.set_ylabel("cm")
    ax.tick_params(axis='x', rotation=0)

plt.suptitle("Moyennes par espèce", fontsize=16)
plt.tight_layout()
plt.show()

# FIGURE 2: Minimums
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    mini = df.groupby("species")[cols[i]].min()
    mini.plot(kind="bar", ax=ax, color="skyblue", label="Min")
    ax.set_title("Minimum " + titres[i])
    ax.set_ylabel("cm")
    ax.tick_params(axis='x', rotation=0)
    ax.legend()

plt.suptitle("Minimums par espèce", fontsize=16)
plt.tight_layout()
plt.show()

# FIGURE 3: Maximums
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    maxi = df.groupby("species")[cols[i]].max()
    maxi.plot(kind="bar", ax=ax, color="salmon", label="Max")
    ax.set_title("Maximum " + titres[i])
    ax.set_ylabel("cm")
    ax.tick_params(axis='x', rotation=0)
    ax.legend()

plt.suptitle("Maximums par espèce", fontsize=16)
plt.tight_layout()
plt.show()
