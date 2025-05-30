from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import winsound
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import entropy
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import hdbscan
import math
from sklearn.model_selection import train_test_split, ParameterGrid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict





path = r"C:\Minamo\Cas_Tuto\Optimization_TQW83E\points.csv"

# Charger le fichier CSV
df = pd.read_csv(path)

# === 1. Affichage des points avec zone rouge ===
def plot_points_with_red_zone(df):
    plt.figure(figsize=(6,6))

    # Points
    plt.scatter(df["X01"], df["X02"], color='blue', alpha=0.6, label='Points')

    # Zone rouge en dehors de [0,1]x[0,1]
    plt.axvspan(-1, 0, color='red', alpha=0.2)
    plt.axvspan(1, max(df["X01"].max(), 1.1), color='red', alpha=0.2)
    plt.axhspan(-1, 0, color='red', alpha=0.2)
    plt.axhspan(1, max(df["X02"].max(), 1.1), color='red', alpha=0.2)

    plt.xlim(min(-0.1, df["X01"].min()), max(1.1, df["X01"].max()))
    plt.ylim(min(-0.1, df["X02"].min()), max(1.1, df["X02"].max()))

    plt.xlabel("X01",fontweight='bold')
    plt.ylabel("X02",fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# === 2. Compte des points hors [0,1] par colonne X01 à X10 ===
def count_out_of_bounds(df):
    count_dict = {}
    for col in [f"X{str(i).zfill(2)}" for i in range(1, 11)]:
        count = ((df[col] < 0) | (df[col] > 1)).sum()
        count_dict[col] = count
    return count_dict

# === 3. Pourcentage de points hors [0,1]^10 ===
def percent_out_of_domain(df):
    in_bounds = df[[f"X{str(i).zfill(2)}" for i in range(1, 11)]].applymap(lambda x: 0 <= x <= 1)
    is_valid = in_bounds.all(axis=1)
    percent_invalid = (~is_valid).sum() / len(df) * 100
    return percent_invalid

# === Utilisation ===
plot_points_with_red_zone(df)

counts = count_out_of_bounds(df)
print("\nNombre de points hors de [0,1] par colonne :")
for k, v in counts.items():
    print(f"{k} : {v}")

percent_invalid = percent_out_of_domain(df)
print(f"\nPourcentage de points hors de [0,1]^10 : {percent_invalid:.2f}%")


## WING
path = r"C:\Minamo\0Low_Wing\data1.csv"

# Charger le fichier CSV
df = pd.read_csv(path)

# === 1. Affichage des points avec zone rouge ===
def plot_points_with_red_zone(df):
    plt.figure(figsize=(6,6))

    # Points
    plt.scatter(df["X1"], df["X2"], color='blue', alpha=0.6, label='Points')

    # Zone rouge en dehors de [0,1]x[0,1]
    plt.axvspan(-1, 0, color='red', alpha=0.2)
    plt.axvspan(1, max(df["X1"].max(), 1.1), color='red', alpha=0.2)
    plt.axhspan(-1, 0, color='red', alpha=0.2)
    plt.axhspan(1, max(df["X2"].max(), 1.1), color='red', alpha=0.2)

    plt.xlim(min(-0.1, df["X1"].min()), max(1.1, df["X1"].max()))
    plt.ylim(min(-0.1, df["X2"].min()), max(1.1, df["X2"].max()))

    plt.xlabel("X1",fontweight='bold')
    plt.ylabel("X2",fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# === 2. Compte des points hors [0,1] par colonne X ===
def count_out_of_bounds(df):
    count_dict = {}
    # On sélectionne dynamiquement les colonnes commençant par 'X'
    x_cols = [col for col in df.columns if col.startswith("X")]
    for col in x_cols:
        count = ((df[col] < 0) | (df[col] > 1)).sum()
        count_dict[col] = count
    return count_dict

# === 3. Pourcentage de points hors de [0,1]^d (d = nombre de colonnes X) ===
def percent_out_of_domain(df):
    x_cols = [col for col in df.columns if col.startswith("X")]
    in_bounds = df[x_cols].applymap(lambda x: 0 <= x <= 1)
    is_valid = in_bounds.all(axis=1)
    percent_invalid = (~is_valid).sum() / len(df) * 100
    return percent_invalid

# === Utilisation ===
plot_points_with_red_zone(df)

counts = count_out_of_bounds(df)
print("\nNombre de points hors de [0,1] par colonne :")
for k, v in counts.items():
    print(f"{k} : {v}")

percent_invalid = percent_out_of_domain(df)
print(f"\nPourcentage de points hors de [0,1]^d : {percent_invalid:.2f}%")




## Wing avec groupement RL

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_best_ids(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "best-ids":
                return value
            else:
                found = find_best_ids(value)
                if found:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_best_ids(item)
            if found:
                return found
    return None

def load_best_individuals(folder_path, label):
    records = []
    for json_file in Path(folder_path).glob("Optimization_*.json"):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except Exception as e:
                print(f"Erreur lecture {json_file.name} : {e}")
                continue

            best_ids = set(find_best_ids(data) or [])
            population = data.get("population", {}).get("points", [])

            for point in population:
                point_id = point.get("ID")
                if point_id in best_ids:
                    records.append({
                        "folder": label,
                        "file": json_file.name,
                        "ID": point_id,
                        "parameters": point.get("parameters", []),
                        "responses": [float(x) for x in point.get("responses", [])]
                    })
    return pd.DataFrame(records)

# Chargement des deux dossiers
df1 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\RL\Data", 
    "RL"
)
df2 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\Data", 
    "Base"
)

# Concaténation
df_all = pd.concat([df1, df2], ignore_index=True)

# Extraction des réponses en colonnes séparées
df_all[["r1", "r2", "r3", "r4"]] = df_all["responses"].apply(lambda r: pd.Series(r))

# -------- PLOTTING -------- #
colors = {"RL": "blue", "Base": "orange"}

# Figure avec deux sous-graphes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# r1 vs r2
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[0].scatter(subset["r1"], subset["r2"], label=label, color=colors[label], alpha=0.7)

axes[0].set_title("Réponse 1 vs Réponse 2")
axes[0].set_xlabel("r1")
axes[0].set_ylabel("r2")
axes[0].legend()

# r3 vs r4
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[1].scatter(subset["r3"], subset["r4"], label=label, color=colors[label], alpha=0.7)

axes[1].set_title("Réponse 3 vs Réponse 4")
axes[1].set_xlabel("r3")
axes[1].set_ylabel("r4")
axes[1].legend()

plt.tight_layout()
plt.show()


import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_best_ids(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "best-ids":
                return value
            else:
                found = find_best_ids(value)
                if found:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_best_ids(item)
            if found:
                return found
    return None

def load_best_individuals(folder_path, label):
    records = []
    for json_file in Path(folder_path).glob("Optimization_*.json"):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except Exception as e:
                print(f"Erreur lecture {json_file.name} : {e}")
                continue

            best_ids = set(find_best_ids(data) or [])
            population = data.get("population", {}).get("points", [])

            for point in population:
                point_id = point.get("ID")
                if point_id in best_ids:
                    records.append({
                        "folder": label,
                        "file": json_file.name,
                        "ID": point_id,
                        "parameters": point.get("parameters", []),
                        "responses": [float(x) for x in point.get("responses", [])]
                    })
    return pd.DataFrame(records)

# Chargement des deux dossiers
df1 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\RL\Data", 
    "RL"
)
df2 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\Data", 
    "Base"
)

# Concaténation
df_all = pd.concat([df1, df2], ignore_index=True)

# Extraction des réponses en colonnes séparées
df_all[["r1", "r2", "r3", "r4"]] = df_all["responses"].apply(lambda r: pd.Series(r))

# -------- PLOTTING -------- #
colors = {"RL": "blue", "Base": "orange"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# r1 vs r4
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[0].scatter(subset["r1"], subset["r4"], label=label, color=colors[label], alpha=0.7)

axes[0].set_title("Réponse 1 vs Réponse 4")
axes[0].set_xlabel("r1")
axes[0].set_ylabel("r4")
axes[0].legend()

# r2 vs r3
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[1].scatter(subset["r2"], subset["r3"], label=label, color=colors[label], alpha=0.7)

axes[1].set_title("Réponse 2 vs Réponse 3")
axes[1].set_xlabel("r2")
axes[1].set_ylabel("r3")
axes[1].legend()

plt.tight_layout()
plt.show()






## Wing avec groupement RF

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_best_ids(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "best-ids":
                return value
            else:
                found = find_best_ids(value)
                if found:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_best_ids(item)
            if found:
                return found
    return None

def load_best_individuals(folder_path, label):
    records = []
    for json_file in Path(folder_path).glob("Optimization_*.json"):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except Exception as e:
                print(f"Erreur lecture {json_file.name} : {e}")
                continue

            best_ids = set(find_best_ids(data) or [])
            population = data.get("population", {}).get("points", [])

            for point in population:
                point_id = point.get("ID")
                if point_id in best_ids:
                    records.append({
                        "folder": label,
                        "file": json_file.name,
                        "ID": point_id,
                        "parameters": point.get("parameters", []),
                        "responses": [float(x) for x in point.get("responses", [])]
                    })
    return pd.DataFrame(records)

# Chargement des deux dossiers
df1 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\RF\Data", 
    "RL"
)
df2 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\Data", 
    "Base"
)

# Concaténation
df_all = pd.concat([df1, df2], ignore_index=True)

# Extraction des réponses en colonnes séparées
df_all[["r1", "r2", "r3", "r4"]] = df_all["responses"].apply(lambda r: pd.Series(r))

# -------- PLOTTING -------- #
colors = {"RL": "blue", "Base": "orange"}

# Figure avec deux sous-graphes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# r1 vs r2
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[0].scatter(subset["r1"], subset["r2"], label=label, color=colors[label], alpha=0.7)

axes[0].set_title("Réponse 1 vs Réponse 2")
axes[0].set_xlabel("r1")
axes[0].set_ylabel("r2")
axes[0].legend()

# r3 vs r4
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[1].scatter(subset["r3"], subset["r4"], label=label, color=colors[label], alpha=0.7)

axes[1].set_title("Réponse 3 vs Réponse 4")
axes[1].set_xlabel("r3")
axes[1].set_ylabel("r4")
axes[1].legend()

plt.tight_layout()
plt.show()









import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_best_ids(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "best-ids":
                return value
            else:
                found = find_best_ids(value)
                if found:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_best_ids(item)
            if found:
                return found
    return None

def load_best_individuals(folder_path, label):
    records = []
    for json_file in Path(folder_path).glob("Optimization_*.json"):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except Exception as e:
                print(f"Erreur lecture {json_file.name} : {e}")
                continue

            best_ids = set(find_best_ids(data) or [])
            population = data.get("population", {}).get("points", [])

            for point in population:
                point_id = point.get("ID")
                if point_id in best_ids:
                    records.append({
                        "folder": label,
                        "file": json_file.name,
                        "ID": point_id,
                        "parameters": point.get("parameters", []),
                        "responses": [float(x) for x in point.get("responses", [])]
                    })
    return pd.DataFrame(records)

# Chargement des deux dossiers
df1 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\RF\Data", 
    "RL"
)
df2 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\Data", 
    "Base"
)

# Concaténation
df_all = pd.concat([df1, df2], ignore_index=True)

# Extraction des réponses en colonnes séparées
df_all[["r1", "r2", "r3", "r4"]] = df_all["responses"].apply(lambda r: pd.Series(r))

# -------- PLOTTING -------- #
colors = {"RL": "blue", "Base": "orange"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# r1 vs r4
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[0].scatter(subset["r1"], subset["r4"], label=label, color=colors[label], alpha=0.7)

axes[0].set_title("Réponse 1 vs Réponse 4")
axes[0].set_xlabel("r1")
axes[0].set_ylabel("r4")
axes[0].legend()

# r2 vs r3
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[1].scatter(subset["r2"], subset["r3"], label=label, color=colors[label], alpha=0.7)

axes[1].set_title("Réponse 2 vs Réponse 3")
axes[1].set_xlabel("r2")
axes[1].set_ylabel("r3")
axes[1].legend()

plt.tight_layout()
plt.show()



## Wing avec groupement IM

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_best_ids(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "best-ids":
                return value
            else:
                found = find_best_ids(value)
                if found:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_best_ids(item)
            if found:
                return found
    return None

def load_best_individuals(folder_path, label):
    records = []
    for json_file in Path(folder_path).glob("Optimization_*.json"):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except Exception as e:
                print(f"Erreur lecture {json_file.name} : {e}")
                continue

            best_ids = set(find_best_ids(data) or [])
            population = data.get("population", {}).get("points", [])

            for point in population:
                point_id = point.get("ID")
                if point_id in best_ids:
                    records.append({
                        "folder": label,
                        "file": json_file.name,
                        "ID": point_id,
                        "parameters": point.get("parameters", []),
                        "responses": [float(x) for x in point.get("responses", [])]
                    })
    return pd.DataFrame(records)

# Chargement des deux dossiers
df1 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\IM\Data", 
    "RL"
)
df2 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\Data", 
    "Base"
)

# Concaténation
df_all = pd.concat([df1, df2], ignore_index=True)

# Extraction des réponses en colonnes séparées
df_all[["r1", "r2", "r3", "r4"]] = df_all["responses"].apply(lambda r: pd.Series(r))

# -------- PLOTTING -------- #
colors = {"RL": "blue", "Base": "orange"}

# Figure avec deux sous-graphes
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# r1 vs r2
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[0].scatter(subset["r1"], subset["r2"], label=label, color=colors[label], alpha=0.7)

axes[0].set_title("Réponse 1 vs Réponse 2")
axes[0].set_xlabel("r1")
axes[0].set_ylabel("r2")
axes[0].legend()

# r3 vs r4
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[1].scatter(subset["r3"], subset["r4"], label=label, color=colors[label], alpha=0.7)

axes[1].set_title("Réponse 3 vs Réponse 4")
axes[1].set_xlabel("r3")
axes[1].set_ylabel("r4")
axes[1].legend()

plt.tight_layout()
plt.show()









import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_best_ids(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "best-ids":
                return value
            else:
                found = find_best_ids(value)
                if found:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_best_ids(item)
            if found:
                return found
    return None

def load_best_individuals(folder_path, label):
    records = []
    for json_file in Path(folder_path).glob("Optimization_*.json"):
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except Exception as e:
                print(f"Erreur lecture {json_file.name} : {e}")
                continue

            best_ids = set(find_best_ids(data) or [])
            population = data.get("population", {}).get("points", [])

            for point in population:
                point_id = point.get("ID")
                if point_id in best_ids:
                    records.append({
                        "folder": label,
                        "file": json_file.name,
                        "ID": point_id,
                        "parameters": point.get("parameters", []),
                        "responses": [float(x) for x in point.get("responses", [])]
                    })
    return pd.DataFrame(records)

# Chargement des deux dossiers
df1 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\IM\Data", 
    "RL"
)
df2 = load_best_individuals(
    r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\Data", 
    "Base"
)

# Concaténation
df_all = pd.concat([df1, df2], ignore_index=True)

# Extraction des réponses en colonnes séparées
df_all[["r1", "r2", "r3", "r4"]] = df_all["responses"].apply(lambda r: pd.Series(r))

# -------- PLOTTING -------- #
colors = {"RL": "blue", "Base": "orange"}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# r1 vs r4
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[0].scatter(subset["r1"], subset["r4"], label=label, color=colors[label], alpha=0.7)

axes[0].set_title("Réponse 1 vs Réponse 4")
axes[0].set_xlabel("r1")
axes[0].set_ylabel("r4")
axes[0].legend()

# r2 vs r3
for label in df_all["folder"].unique():
    subset = df_all[df_all["folder"] == label]
    axes[1].scatter(subset["r2"], subset["r3"], label=label, color=colors[label], alpha=0.7)

axes[1].set_title("Réponse 2 vs Réponse 3")
axes[1].set_xlabel("r2")
axes[1].set_ylabel("r3")
axes[1].legend()

plt.tight_layout()
plt.show()




## Trouver l'optimum avec toutes les variables

import json

path = r"C:\Minamo\0Low_Wing\Optimization_GNSSKI\Optimization_GNSSKI.O.json"

# Charger le fichier JSON
with open(path, "r") as f:
    data = json.load(f)

# Extraire la population
population = data["population"]["points"]

# Initialiser la meilleure solution
best_point = None
best_value = float("inf")

# Parcourir les individus de la population
for point in population:
    if point.get("global-success", False):
        value = point.get("global-objective", float("inf"))
        if value < best_value:
            best_value = value
            best_point = point

# Afficher l'optimum et créer la heatmap
if best_point:
    print(">>> Meilleur individu trouvé :")
    print(f"ID : {best_point['ID']}")
    print(f"Valeur de l'objectif : {best_value}")
    
    modified_params = []
    print("Paramètres modifiés :")
    for name, val in zip(data["psk"]["parameters"], best_point["parameters"]):
        param_name = name['name']
        param_value = float(val)
        if param_value > 1:
            param_value = 1
        modified_params.append(param_value)
        print(f"  {param_name} = {param_value}")

    print("Réponses :")
    for name, val in zip(data["osk"]["outputs"], best_point["responses"]):
        print(f"  {name['name']} = {val}")

    # --- Heatmap ---
    heatmap = np.array([modified_params])  # 1 ligne
    plt.figure(figsize=(12, 1.5))
    plt.imshow(heatmap, cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Intensité (0 = blanc, 1 = noir)", orientation="horizontal")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()


else:
    print("Aucun individu valide trouvé.")



## Trouver l'optimum avec un groupe restreint RL

import json
import matplotlib.pyplot as plt
import numpy as np

# === FICHIERS ===
opt_path = r"C:\Minamo\0WingRL\Optimization_H2J9YH\Optimization_H2J9YH.O.json"
cluster_path = r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\RL\wing_variables_par_cluster.txt"

# === CHARGER LES DONNÉES ===
with open(opt_path, "r") as f:
    data = json.load(f)

with open(cluster_path, "r", encoding="utf-8") as f:
    clusters = [list(map(int, line.strip().split())) for line in f]

# === TROUVER LE MEILLEUR INDIVIDU ===
population = data["population"]["points"]
best_point = None
best_value = float("inf")

for point in population:
    if point.get("global-success", False):
        value = point.get("global-objective", float("inf"))
        if value < best_value:
            best_value = value
            best_point = point

# === TRAITEMENT ET AFFICHAGE HEATMAP ===
if best_point:
    print(">>> Meilleur individu trouvé :")
    print(f"ID : {best_point['ID']}")
    print(f"Valeur de l'objectif : {best_value}")

    # Étape 1 : extraire les paramètres et les corriger (>1 → 1)
    compressed_paramsRL = []
    print("Paramètres modifiés :")
    for name, val in zip(data["psk"]["parameters"], best_point["parameters"]):
        param_name = name['name']
        param_value = float(val)
        if param_value > 1:
            param_value = 1
        compressed_paramsRL.append(param_value)
        print(f"  {param_name} = {param_value}")

    # Étape 2 : décompresser selon les clusters
    max_index = max(idx for cluster in clusters for idx in cluster)
    decompressed_paramsRL = [0.0] * (max_index + 1)  # ex: 162 pour dome

    for cluster_value, indices in zip(compressed_paramsRL, clusters):
        for idx in indices:
            decompressed_paramsRL[idx] = cluster_value

    print("Réponses :")
    for name, val in zip(data["osk"]["outputs"], best_point["responses"]):
        print(f"  {name['name']} = {val}")

    # Étape 3 : afficher la heatmap 1D
    heatmap = np.array([decompressed_paramsRL])
    plt.figure(figsize=(15, 1.5))
    plt.imshow(heatmap, cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Intensité (0 = blanc, 1 = noir)", orientation="horizontal")
    plt.xticks([])  # tu peux mettre range(len(...)) si tu veux les indices
    plt.yticks([])
    plt.title("Heatmap des paramètres décompressés par cluster")
    plt.tight_layout()
    plt.show()

else:
    print("Aucun individu valide trouvé.")


## Trouver l'optimum avec un groupe restreint RF

import json
import matplotlib.pyplot as plt
import numpy as np

# === FICHIERS ===
opt_path = r"C:\Minamo\0WingRF\Optimization_D806OT\Optimization_D806OT.O.json"
cluster_path = r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\RF\wing_variables_par_cluster.txt"

# === CHARGER LES DONNÉES ===
with open(opt_path, "r") as f:
    data = json.load(f)

with open(cluster_path, "r", encoding="utf-8") as f:
    clusters = [list(map(int, line.strip().split())) for line in f]

# === TROUVER LE MEILLEUR INDIVIDU ===
population = data["population"]["points"]
best_point = None
best_value = float("inf")

for point in population:
    if point.get("global-success", False):
        value = point.get("global-objective", float("inf"))
        if value < best_value:
            best_value = value
            best_point = point

# === TRAITEMENT ET AFFICHAGE HEATMAP ===
if best_point:
    print(">>> Meilleur individu trouvé :")
    print(f"ID : {best_point['ID']}")
    print(f"Valeur de l'objectif : {best_value}")

    # Étape 1 : extraire les paramètres et les corriger (>1 → 1)
    compressed_paramsRF = []
    print("Paramètres modifiés :")
    for name, val in zip(data["psk"]["parameters"], best_point["parameters"]):
        param_name = name['name']
        param_value = float(val)
        if param_value > 1:
            param_value = 1
        compressed_paramsRF.append(param_value)
        print(f"  {param_name} = {param_value}")

    # Étape 2 : décompresser selon les clusters
    max_index = max(idx for cluster in clusters for idx in cluster)
    decompressed_paramsRF = [0.0] * (max_index + 1)  # ex: 162 pour dome

    for cluster_value, indices in zip(compressed_paramsRF, clusters):
        for idx in indices:
            decompressed_paramsRF[idx] = cluster_value

    print("Réponses :")
    for name, val in zip(data["osk"]["outputs"], best_point["responses"]):
        print(f"  {name['name']} = {val}")

    # Étape 3 : afficher la heatmap 1D
    heatmap = np.array([decompressed_paramsRF])
    plt.figure(figsize=(15, 1.5))
    plt.imshow(heatmap, cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Intensité (0 = blanc, 1 = noir)", orientation="horizontal")
    plt.xticks([])  # tu peux mettre range(len(...)) si tu veux les indices
    plt.yticks([])
    plt.title("Heatmap des paramètres décompressés par cluster")
    plt.tight_layout()
    plt.show()

else:
    print("Aucun individu valide trouvé.")




## Trouver l'optimum avec un groupe restreint IM

import json
import matplotlib.pyplot as plt
import numpy as np

# === FICHIERS ===
opt_path = r"C:\Minamo\0WingIM\Optimization_MIDVQZ\Optimization_MIDVQZ.O.json"
cluster_path = r"C:\Users\liamh\OneDrive - Université de Namur\Memoire\Figures\Groupement\IM\wing_variables_par_cluster.txt"

# === CHARGER LES DONNÉES ===
with open(opt_path, "r") as f:
    data = json.load(f)

with open(cluster_path, "r", encoding="utf-8") as f:
    clusters = [list(map(int, line.strip().split())) for line in f]

# === TROUVER LE MEILLEUR INDIVIDU ===
population = data["population"]["points"]
best_point = None
best_value = float("inf")

for point in population:
    if point.get("global-success", False):
        value = point.get("global-objective", float("inf"))
        if value < best_value:
            best_value = value
            best_point = point

# === TRAITEMENT ET AFFICHAGE HEATMAP ===
if best_point:
    print(">>> Meilleur individu trouvé :")
    print(f"ID : {best_point['ID']}")
    print(f"Valeur de l'objectif : {best_value}")

    # Étape 1 : extraire les paramètres et les corriger (>1 → 1)
    compressed_paramsIM = []
    print("Paramètres modifiés :")
    for name, val in zip(data["psk"]["parameters"], best_point["parameters"]):
        param_name = name['name']
        param_value = float(val)
        if param_value > 1:
            param_value = 1
        compressed_paramsIM.append(param_value)
        print(f"  {param_name} = {param_value}")

    # Étape 2 : décompresser selon les clusters
    max_index = max(idx for cluster in clusters for idx in cluster)
    decompressed_paramsIM = [0.0] * (max_index + 1)  # ex: 162 pour dome

    for cluster_value, indices in zip(compressed_paramsIM, clusters):
        for idx in indices:
            decompressed_paramsIM[idx] = cluster_value

    print("Réponses :")
    for name, val in zip(data["osk"]["outputs"], best_point["responses"]):
        print(f"  {name['name']} = {val}")

    # Étape 3 : afficher la heatmap 1D
    heatmap = np.array([decompressed_paramsIM])
    plt.figure(figsize=(15, 1.5))
    plt.imshow(heatmap, cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Intensité (0 = blanc, 1 = noir)", orientation="horizontal")
    plt.xticks([])  # tu peux mettre range(len(...)) si tu veux les indices
    plt.yticks([])
    plt.title("Heatmap des paramètres décompressés par cluster")
    plt.tight_layout()
    plt.show()

else:
    print("Aucun individu valide trouvé.")


# Affichage côte à côte


import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def afficher_heatmaps_individuelles(original, decompresses, methodes, save_dir="../../Figures/Opti/Wing/"):
    """
    Affiche une figure par méthode avec l'original en haut et la version réduite en bas.

    - original : heatmap d'origine (identique pour tous)
    - decompresses : liste des heatmaps après réduction (dans le même ordre que `methodes`)
    - methodes : liste des noms des méthodes ["RL", "RF", "IM"]
    - save_dir : répertoire de sauvegarde
    """
    for methode, decompressee in zip(methodes, decompresses):
        fig, axs = plt.subplots(2, 1, figsize=(4, 3.6), squeeze=False)

        # Original
        axs[0][0].imshow(np.array([original]), cmap="gray_r", aspect="auto", vmin=0, vmax=1)
        axs[0][0].set_title(f"Sans Groupement", fontsize=12)
        axs[0][0].set_xticks([])
        axs[0][0].set_yticks([])

        # Décompressée
        axs[1][0].imshow(np.array([decompressee]), cmap="gray_r", aspect="auto", vmin=0, vmax=1)
        axs[1][0].set_title(f"Avec Groupement ({methode})", fontsize=12)
        axs[1][0].set_xticks([])
        axs[1][0].set_yticks([])

        # Colorbar
        fig.subplots_adjust(right=0.8, bottom=0.2)
        cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])
        fig.colorbar(axs[1][0].images[0], cax=cbar_ax, orientation='horizontal', label="Intensité (0 = blanc, 1 = noir)")

        plt.tight_layout(rect=[0, 0.1, 1, 1])

        # Sauvegarde
        nom_fichier = f"{save_dir}heatmap_{methode}.pdf"
        plt.savefig(nom_fichier, dpi=500, bbox_inches='tight')
        plt.show()

original = modified_params
decompresses = [decompressed_paramsRL, decompressed_paramsRF, decompressed_paramsIM]
methodes = ["RL", "RF", "IM"]

afficher_heatmaps_individuelles(original, decompresses, methodes)


## R^2
def plot_prediction_scatter(y_true, y_pred, save_path=None):
    """
    Affiche un nuage de points (réel vs prédiction) avec métriques.
    
    Paramètres :
        - y_true : valeurs cibles vraies
        - y_pred : prédictions du modèle
        - save_path : chemin de sauvegarde éventuel
    """
    # Calcul des métriques
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    corr, p_value = pearsonr(y_true.ravel(), y_pred.ravel())

    print("\n--- Évaluation ---")
    print(f"R² score : {r2:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"Corrélation de Pearson : {corr:.4f} (p-value = {p_value:.4e})")

    # Affichage du scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=1, color='blue', edgecolor='k')    
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label=f"R² = {r2:.4f}")
    plt.xlabel("Valeurs réelles", fontweight='bold')
    plt.ylabel("Prédictions", fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    bold_font = font_manager.FontProperties(size=18, weight='bold')
    plt.legend(prop=bold_font)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

y_true = np.array(modified_params).flatten()
y_pred = np.array(decompressed_paramsRF).flatten()

plot_prediction_scatter(y_true, y_pred, "../../Figures/Opti/Wing/R2_RF.pdf")

y_true = np.array(modified_params).flatten()
y_pred = np.array(decompressed_paramsIM).flatten()

mse = mean_squared_error(y_true, y_pred)
print(f"MSE : {mse:.4f}")