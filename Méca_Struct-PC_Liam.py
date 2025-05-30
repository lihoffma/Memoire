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

winsound.Beep(350, 2000)
winsound.Beep(1000, 2000)  

def plot_importance(importance_array, feature_names, save_path=None):
    """
    Affiche l'importance des variables à partir d'un tableau d'importances.
    
    Paramètres :
        - importance_array : array-like, importances (ex: |coef|, feature_importance)
        - feature_names : liste des noms des colonnes
        - save_path : chemin de sauvegarde éventuel
    """
    importance_array = np.sort(importance_array)
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance_array, color='blue')
    plt.ylabel("Importance", fontweight='bold')
    plt.xlabel("Variables", fontweight='bold')
    plt.xticks([])
    plt.yticks(fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

    print("\n--- Importance des variables ---")
    for feature, imp in zip(feature_names, importance_array):
        print(f"{feature}: {imp:.4f}")


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
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue', edgecolor='k')    
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


def plot_histogram(data, save_path=None):
    """
    Affiche un histogramme avec courbe de densité pour une seule variable.

    Paramètres :
        - data : array-like (pandas Series ou numpy array)
        - title : titre du graphique (optionnel)
        - xlabel : label de l’axe x (optionnel)
        - save_path : chemin pour sauvegarder l’image (optionnel)
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data, kde=False, bins=30, color='blue', edgecolor='k')
    plt.xlabel('Valeur', fontweight='bold')
    plt.ylabel("Fréquence", fontweight='bold')
    plt.xticks(rotation=45, fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=500)
    plt.show()

# Histogrammes
cas = "wing"

entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in data_full.columns]]
target = target_full[[str(i) for i in target_full.columns]]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

for i in data.columns:
    plot_histogram(data[f"{i}"], save_path=f"../../Figures/Histogrammes/Structures/{cas}_entree_{i}.pdf")
for j in target.columns:
    plot_histogram(target[f"{j}"], save_path=f"../../Figures/Histogrammes/Structures/{cas}_sortie_{j}.pdf")





# ACP
def ACP(model, X_train, X_test, y_train, y_test, save_prefix):
    """
    Applique une ACP, affiche les résultats, entraîne un modèle de régression et évalue les performances.

    Paramètres :
        - model : modèle sklearn (ex: LinearRegression())
        - X_train : données d'entraînement
        - X_test : données de test
        - y_train : cibles d'entraînement
        - y_test : cibles de test
        - save_prefix : chemin où sauvegarder les figures
    """
    # === Analyse en Composantes Principales (ACP) ===
    pca = PCA()
    X_pca = pca.fit_transform(X_train)
    explained_var = pca.explained_variance_ratio_

    importance = np.abs(pca.components_.T @ explained_var)

    print("\nVariance expliquée par chaque composante principale :")
    sum = 0
    for i, var in enumerate(explained_var):
        # print(f"PC{i+1}: {var:.4f}")
        sum += var
        if sum >= 0.95:
            print(f"95% de la variance expliquée par {i+1} composantes principales.")
            break

    # === Importance des variables ===
    plot_importance(importance, X_train.columns, save_path=f"{save_prefix}acp1.pdf")

    # === Scatter des deux premières composantes ===
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train.values.ravel(), cmap='viridis', edgecolor='k')
    plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)", fontweight='bold')
    plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)", fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    cb = plt.colorbar(scatter)
    for t in cb.ax.get_yticklabels():
        t.set_fontweight('bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}acp2.pdf", dpi=500)
    plt.show()

    # === Régression avec toutes les variables ===
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # === Prédiction ===
    y_pred = lr.predict(X_test)
    plot_prediction_scatter(y_test.values, y_pred, save_path=f"{save_prefix}acp3.pdf")

    # === Régression avec les deux variables les plus importantes ===
    top2_idx = np.argsort(importance)[-(math.ceil(len(X_train.columns)/5)):]
    top2_features = X_train.columns[top2_idx]

    print(f"\nDeux variables les plus importantes : {list(top2_features)}")

    X_train_top2 = X_train[top2_features]
    X_test_top2 = X_test[top2_features]

    lr.fit(X_train_top2, y_train)
    y_pred_top2 = lr.predict(X_test_top2)

    plot_prediction_scatter(y_test.values, y_pred_top2, save_path=f"{save_prefix}acp4.pdf")


cas = "bldg"
# i = target.columns[0]
cas_array = ["wing", "dome", "bldg"]

for cas in cas_array:
    
    entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)

    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    for i in target.columns:
        X_train, X_test, y_train, y_test = train_test_split(data, target[f"{i}"], test_size=0.2, random_state=42)
        ACP(model, X_train, X_test, y_train, y_test, save_prefix=f"../../Figures/Sans Groupement/ACP/{cas}_{i}_")

# RL
def RL(model, X_train, X_test, y_train, y_test, save_prefix):
    """
    Applique une ACP, affiche les résultats, entraîne un modèle de régression et évalue les performances.

    Paramètres :
        - model : modèle sklearn (ex: LinearRegression())
        - X_train : données d'entraînement
        - X_test : données de test
        - y_train : cibles d'entraînement
        - y_test : cibles de test
        - save_prefix : chemin où sauvegarder les figures
    """
    # === Analyse en Composantes Principales (ACP) ===
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    importance = np.abs(lr.coef_.ravel())

    # === Importance des variables ===
    plot_importance(importance, X_train.columns, save_path=f"{save_prefix}rl1.pdf")

    # === Prédiction ===
    y_pred = lr.predict(X_test)
    plot_prediction_scatter(y_test.values, y_pred, save_path=f"{save_prefix}rl2.pdf")

    # === Régression avec le cinquième variables les plus importantes ===
    top2_idx = np.argsort(importance)[-(math.ceil(len(X_train.columns)/5)):]
    top2_features = X_train.columns[top2_idx]

    print(f"\nVariables les plus importantes : {list(top2_features)}")

    X_train_top2 = X_train[top2_features]
    X_test_top2 = X_test[top2_features]

    lr.fit(X_train_top2, y_train)
    y_pred_top2 = lr.predict(X_test_top2)

    plot_prediction_scatter(y_test.values, y_pred_top2, save_path=f"{save_prefix}rl3.pdf")

cas = "wing"
cas_array = ["wing", "dome", "bldg"]
i = target.columns[0]

for cas in cas_array:
    
    entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)

    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    for i in target.columns:
        X_train, X_test, y_train, y_test = train_test_split(data, target[f"{i}"], test_size=0.2, random_state=42)
        RL(model, X_train, X_test, y_train, y_test, save_prefix=f"../../Figures/Sans Groupement/RL/{cas}_{i}_")

cas = "bldg"
entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in data_full.columns]]
target = target_full[[str(i) for i in target_full.columns]]
i = target.columns[0]
X_train, X_test, y_train, y_test = train_test_split(data, target[f"{i}"], test_size=0.2, random_state=42)


# === Régression linéaire et importance ===
lr = LinearRegression()
lr.fit(X_train, y_train)
importance = np.abs(lr.coef_.ravel())

# === Création d'un DataFrame pour la lisibilité ===
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=True)

# === Détermination du nombre optimal de clusters (coude + silhouette) ===
X_importance = importance_df[['Importance']].values
inertia = []
silhouette_scores = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_importance)
    inertia.append(kmeans.inertia_)
    
    if k > 1:
        silhouette_scores.append(silhouette_score(X_importance, kmeans.labels_))
    else:
        silhouette_scores.append(np.nan)

# 📈 Graphe du coude
plt.figure(figsize=(10, 4))
plt.plot(K_range, inertia, marker='o', color='blue')
plt.xlabel("Nombre de clusters", fontweight='bold')
plt.ylabel("Inertie", fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(True)
plt.savefig("../../Figures/Sans Groupement/RL/bldg_kmeans1.pdf", dpi=500)
plt.show()

# 📈 Graphe du score de silhouette
plt.figure(figsize=(10, 4))
plt.plot(K_range[1:], silhouette_scores[1:], marker='o', color='red')
plt.xlabel("Nombre de clusters", fontweight='bold')
plt.ylabel("Score de silhouette", fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.grid(True)
plt.savefig("../../Figures/Sans Groupement/RL/bldg_kmeans2.pdf", dpi=500)
plt.show()

# ✅ Choix optimal du nombre de clusters
optimal_k = np.argmax(silhouette_scores[1:]) + 2  # +2 car on ignore k=1
print(f"Nombre optimal de clusters estimé : {optimal_k}")

# 🧠 Clustering final
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
importance_df['Cluster'] = kmeans.fit_predict(X_importance)

# Création d'une palette de couleurs distinctes
palette = sns.color_palette("husl", optimal_k)

# 📊 Visualisation avec plt.bar
plt.figure(figsize=(12, 6))

# Tri des variables pour un affichage lisible
sorted_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Association de chaque cluster à sa couleur
colors = [palette[cluster] for cluster in sorted_df['Cluster']]

# Barplot avec couleur selon le cluster
plt.bar(sorted_df['Feature'], sorted_df['Importance'], color=colors)
plt.xticks([], fontweight='bold')
plt.yticks(fontweight='bold')
plt.xlabel("Variables", fontweight='bold')
plt.ylabel("Importance absolue", fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("../../Figures/Sans Groupement/RL/bldg_kmeans3.pdf", dpi=500)
plt.show()

# Regrouper les features par cluster (idem)
clusters = {}
for _, row in importance_df.iterrows():
    cluster_id = row['Cluster']
    feature = row['Feature']
    clusters.setdefault(cluster_id, []).append(feature)

# Obtenir la liste ordonnée des clusters selon leur première apparition dans sorted_df
ordered_clusters = []
for c in sorted_df['Cluster']:
    if c not in ordered_clusters:
        ordered_clusters.append(c)

# Écriture dans le fichier selon cet ordre
with open("../../Figures/Sans Groupement/RL/clusters_variables.txt", "w") as f:
    for cluster_id in ordered_clusters:
        feature_list = clusters[cluster_id]
        feature_list_int = [int(feat) for feat in feature_list]
        f.write(f"{feature_list_int}\n")


# Récupérer la liste ordonnée des clusters selon l'ordre d'apparition dans sorted_df
ordered_clusters = []
for c in sorted_df['Cluster']:
    if c not in ordered_clusters:
        ordered_clusters.append(c)

# Boucle sur ces clusters dans l'ordre d'apparition
i=1
for cluster_focus in ordered_clusters:
    colors = ['red' if c == cluster_focus else 'black' for c in sorted_df['Cluster']]
    
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_df['Feature'], sorted_df['Importance'], color=colors)
    plt.xticks([], fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xlabel("Variables", fontweight='bold')
    plt.ylabel("Importance absolue", fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    filename = f"../../Figures/Sans Groupement/RL/bldg_groupe{i}.pdf"
    plt.savefig(filename, dpi=500)
    plt.show()
    i+=1
    plt.close()

# Random Forest
def RF(model, X_train, X_test, y_train, y_test, save_prefix):
    """
    Applique une ACP, affiche les résultats, entraîne un modèle de régression et évalue les performances.

    Paramètres :
        - model : modèle sklearn (ex: LinearRegression())
        - X_train : données d'entraînement
        - X_test : données de test
        - y_train : cibles d'entraînement
        - y_test : cibles de test
        - save_prefix : chemin où sauvegarder les figures
    """
    # Modèle Random Forest initial
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.values.ravel())

    importance = rf.feature_importances_

    # === Importance des variables ===
    plot_importance(importance, X_train.columns, save_path=f"{save_prefix}rf1.pdf")

    # === Prédiction ===
    y_pred = rf.predict(X_test)
    plot_prediction_scatter(y_test.values, y_pred, save_path=f"{save_prefix}rf2.pdf")

    # === Régression avec le cinquième variables les plus importantes ===
    top2_idx = np.argsort(importance)[-(math.ceil(len(X_train.columns)/5)):]
    top2_features = X_train.columns[top2_idx]

    print(f"\nVariables les plus importantes : {list(top2_features)}")

    X_train_top2 = X_train[top2_features]
    X_test_top2 = X_test[top2_features]

    rf.fit(X_train_top2, y_train.values.ravel())
    y_pred_top2 = rf.predict(X_test_top2)

    plot_prediction_scatter(y_test.values, y_pred_top2, save_path=f"{save_prefix}rf3.pdf")

cas = "bldg"
cas_array = ["bldg"]

for cas in cas_array:
    
    entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)

    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    for i in tqdm(target.columns):
        X_train, X_test, y_train, y_test = train_test_split(data, target[f"{i}"], test_size=0.2, random_state=42)
        RF(model, X_train, X_test, y_train, y_test, save_prefix=f"../../Figures/Sans Groupement/RF/{cas}_{i}_")


# Information mutuelle
def IM(model, X_train, X_test, y_train, y_test, save_prefix):
    """
    Applique une ACP, affiche les résultats, entraîne un modèle de régression et évalue les performances.

    Paramètres :
        - model : modèle sklearn (ex: LinearRegression())
        - X_train : données d'entraînement
        - X_test : données de test
        - y_train : cibles d'entraînement
        - y_test : cibles de test
        - save_prefix : chemin où sauvegarder les figures
    """
    # Modèle information mutuelle

    importance = mutual_info_regression(X_train, y_train.values.ravel(), random_state=42)
    mi_series = pd.Series(importance, index=data.columns)

    # === Importance des variables ===
    plot_importance(importance, X_train.columns, save_path=f"{save_prefix}im1.pdf")

    # === Prédiction ===
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    plot_prediction_scatter(y_test.values, y_pred, save_path=f"{save_prefix}im2.pdf")

    # === Régression avec le cinquième variables les plus importantes ===
    top2_idx = np.argsort(importance)[-(math.ceil(len(X_train.columns)/5)):]
    top2_features = X_train.columns[top2_idx]

    print(f"\nVariables les plus importantes : {list(top2_features)}")

    X_train_top2 = X_train[top2_features]
    X_test_top2 = X_test[top2_features]

    lr.fit(X_train_top2, y_train.values.ravel())
    y_pred_top2 = lr.predict(X_test_top2)

    plot_prediction_scatter(y_test.values, y_pred_top2, save_path=f"{save_prefix}im3.pdf")

cas = "wing"
cas_array = ["wing", "dome", "bldg"]

for cas in cas_array:
    
    entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)

    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    for i in target.columns:
        X_train, X_test, y_train, y_test = train_test_split(data, target[f"{i}"], test_size=0.2, random_state=42)
        IM(model, X_train, X_test, y_train, y_test, save_prefix=f"../../Figures/Sans Groupement/IM/{cas}_{i}_")


# Influence du nombre de voisins sur l'information mutuelle

cas = "wing"
entrees = choix(cas)[0]     
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in data_full.columns]]
target = target_full[[str(i) for i in target_full.columns]]

i = target.columns[0]
for i in target.columns:
    X_train, X_test, y_train, y_test = train_test_split(data, target[f"{i}"], test_size=0.2, random_state=42)
    for j in [3,5,10,50,100,200,400,800]:
        # Modèle information mutuelle
        importance = mutual_info_regression(X_train, y_train.values.ravel(), random_state=42, n_neighbors=j)
        mi_series = pd.Series(importance, index=data.columns)

        # === Importance des variables ===
        plot_importance(importance, X_train.columns, save_path=f"../../Figures/Problèmes identifiés/{cas}_{i}_im_N_{j}.pdf")


# Problèmes identifiés : information mutuelle et le nombre de voisins Perte de précision

# === Paramètres ===
cas = "wing"
entrees = choix(cas)[0]
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)
data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in data_full.columns]]
target = target_full[[str(i) for i in target_full.columns]]

percent_samples = [20, 40, 60, 80, 100]
ratios_kN = [0.05, 0.1, 0.16, 0.2, 0.3, 0.4, 0.5]

results_r2 = {}
results_nonzero = {}

# === Calcul principal ===
for col in target.columns:
    for p_samp in tqdm(percent_samples, desc="Taille échantillon train"):
        n_samples = int(len(data) * (p_samp / 100))
        X_sub = data.iloc[:n_samples]
        Y_sub = target.iloc[:n_samples]

        X_train, X_test, y_train, y_test = train_test_split(X_sub, Y_sub[col], test_size=0.2, random_state=42)

        model_all = LinearRegression()
        model_all.fit(X_train, y_train)
        y_pred_all = model_all.predict(X_test)
        r2_full = r2_score(y_test, y_pred_all)

        pertes_k = []
        nonzero_counts = []
        effective_ratios = []

        for ratio in ratios_kN:
            k = max(1, math.ceil(len(X_train) * ratio))
            if k >= len(X_train):
                continue
            effective_ratios.append(k / len(X_train))

            importance = mutual_info_regression(X_train, y_train.values.ravel(), random_state=42, n_neighbors=k)
            mi_df = pd.DataFrame({'Feature': X_train.columns, 'MI': importance})
            mi_df = mi_df.sort_values(by='MI', ascending=False).reset_index(drop=True)

            n_top = max(1, int(0.2 * len(mi_df)))
            top_features = mi_df.iloc[:n_top]['Feature'].tolist()

            # R²
            model_sub = LinearRegression()
            model_sub.fit(X_train[top_features], y_train)
            y_pred_sub = model_sub.predict(X_test[top_features])
            r2_sub = r2_score(y_test, y_pred_sub)

            perte = (1 - (r2_sub / r2_full)) * 100 if r2_full != 0 else 100
            pertes_k.append(perte)

            # Nombre de features informatives (MI > 0)
            n_nonzero = np.sum(importance > 0)
            nonzero_counts.append(n_nonzero)

        results_r2[p_samp] = (effective_ratios, pertes_k)
        results_nonzero[p_samp] = (effective_ratios, nonzero_counts)

    # === Graphique 1 : Perte en R²
    plt.figure(figsize=(10, 5))
    for p_samp in percent_samples:
        if p_samp in results_r2:
            ratios, pertes = results_r2[p_samp]
            plt.plot(ratios, pertes, marker='o', label=f'{p_samp}% échantillon')
    plt.axvline(0.16, color='red', linestyle='--', label='k/N = 0.16')
    plt.xlabel("Ratio k/N", fontweight='bold')
    plt.ylabel("Perte relative en R² (%)", fontweight='bold')
    plt.legend(fontsize=18, prop=font_manager.FontProperties(weight='bold'))
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True)
    plt.savefig(f"../../Figures/Problèmes identifiés/wing_{col}_perte.pdf", dpi=500)
    plt.tight_layout()
    plt.show()

    # === Graphique 2 : Nombre de MI > 0
    plt.figure(figsize=(10, 5))
    for p_samp in percent_samples:
        if p_samp in results_nonzero:
            ratios, n_nonzero = results_nonzero[p_samp]
            plt.plot(ratios, n_nonzero, marker='o', label=f'{p_samp}% échantillon')
    plt.axvline(0.16, color='red', linestyle='--', label='k/N = 0.16')
    plt.xlabel("Ratio k/N", fontweight='bold')
    plt.ylabel("Nombre de features avec IM > 0", fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.legend(fontsize=18, prop=font_manager.FontProperties(weight='bold'))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../Figures/Problèmes identifiés/wing_{col}_IM.pdf", dpi=500)
    plt.show()


# Moyenne

# === Paramètres ===
cas = "wing"
entrees = choix(cas)[0]
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)
data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in data_full.columns]]
target = target_full[[str(i) for i in target_full.columns]]

percent_samples = [20, 40, 60, 80, 100]
ratios_kN = [0.05, 0.1, 0.16, 0.2, 0.3, 0.4, 0.5]



# === Calcul principal ===
for col in target.columns:
    mean_r2_by_ratio = {}
    mean_nonzero_by_ratio = {}

    for ratio in tqdm(ratios_kN, desc=f"Traitement des ratios pour la sortie {col}"):
        pertes = []
        n_nonzero_all = []

        for p_samp in percent_samples:
            n_samples = int(len(data) * (p_samp / 100))
            X_sub = data.iloc[:n_samples]
            Y_sub = target.iloc[:n_samples]

            X_train, X_test, y_train, y_test = train_test_split(X_sub, Y_sub[col], test_size=0.2, random_state=42)

            # Modèle avec toutes les features
            model_all = LinearRegression()
            model_all.fit(X_train, y_train)
            y_pred_all = model_all.predict(X_test)
            r2_full = r2_score(y_test, y_pred_all)

            k = math.ceil(len(X_train) * ratio)
            if k >= len(X_train):
                continue

            importance = mutual_info_regression(X_train, y_train.values.ravel(), random_state=42, n_neighbors=k)
            mi_df = pd.DataFrame({'Feature': X_train.columns, 'MI': importance})
            mi_df = mi_df.sort_values(by='MI', ascending=False).reset_index(drop=True)

            n_top = max(1, int(0.2 * len(mi_df)))
            top_features = mi_df.iloc[:n_top]['Feature'].tolist()

            model_sub = LinearRegression()
            model_sub.fit(X_train[top_features], y_train)
            y_pred_sub = model_sub.predict(X_test[top_features])
            r2_sub = r2_score(y_test, y_pred_sub)

            perte = (1 - (r2_sub / r2_full)) * 100 if r2_full != 0 else 100
            pertes.append(perte)

            n_nonzero = np.sum(importance > 0)
            n_nonzero_all.append(n_nonzero)

        ratio_effectif = k / len(X_train)
        mean_r2_by_ratio[ratio_effectif] = np.mean(pertes)
        mean_nonzero_by_ratio[ratio_effectif] = np.mean(n_nonzero_all)

    # === Graphique 1 : Moyenne de la Perte en R² ===
    plt.figure(figsize=(10, 5))
    ratios = sorted(mean_r2_by_ratio.keys())
    pertes_moy = [mean_r2_by_ratio[r] for r in ratios]
    plt.plot(ratios, pertes_moy, marker='o')
    plt.axvline(0.16, color='red', linestyle='--', label='k/N = 0.16')
    plt.xlabel("Ratio k/N", fontweight='bold')
    plt.ylabel("Moyenne de la Perte en R² (%)", fontweight='bold')
    plt.legend(fontsize=18, prop=font_manager.FontProperties(weight='bold'))
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../Figures/Problèmes identifiés/wing_{col}_moyenne_perte_par_ratio.pdf", dpi=500)
    plt.show()

    # === Graphique 2 : Moyenne du Nombre de MI > 0 ===
    plt.figure(figsize=(10, 5))
    nonzero_moy = [mean_nonzero_by_ratio[r] for r in ratios]
    plt.plot(ratios, nonzero_moy, marker='o')
    plt.axvline(0.16, color='red', linestyle='--', label='k/N = 0.16')
    plt.xlabel("Ratio k/N", fontweight='bold')
    plt.ylabel("Moyenne du Nombre de MI > 0", fontweight='bold')
    plt.legend(fontsize=18, prop=font_manager.FontProperties(weight='bold'))
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"../../Figures/Problèmes identifiés/wing_{col}_moyenne_IM_par_ratio.pdf", dpi=500)
    plt.show()


# ==============================================================================
# Analyse de stabilité de la sélection de variables par information mutuelle
# ------------------------------------------------------------------------------ 
# Ce script évalue la robustesse des top-k variables sélectionnées par 
# information mutuelle en fonction de :
#   - la taille de l'échantillon (20% à 100%)
#   - le ratio k/N utilisé pour le calcul des voisins dans mutual_info_regression
#
# Pour chaque combinaison (échantillon, ratio), on calcule les top-k features.
# On mesure ensuite la distance de Jaccard entre chaque sélection et une 
# sélection de référence (même échantillon, ratio de référence k/N = 0.16).
#
# Une heatmap finale affiche les distances de Jaccard, où une faible valeur 
# indique une sélection stable par rapport à la référence.
# ==============================================================================

# === Paramètres ===
cas = "wing"
entrees = choix(cas)[0]
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)
data = model.donnees(cas)
target = model.sortie(cas)

percent_samples = [20, 40, 60, 80, 100]
ratios_kN = [0.05, 0.1, 0.16, 0.2, 0.3, 0.4, 0.5]
ref_ratio = 0.16
top_k = 20

# === Stocker les top-k features
jaccard_data = {}  # {(p_samp, ratio): [top-k features]}
col = target.columns[3]
# for col in target.columns:
for p_samp in tqdm(percent_samples, desc=f"Échantillons pour {col}"):
    n = int(len(data) * p_samp / 100)
    X_sub = data.iloc[:n]
    y_sub = target[col].iloc[:n]
    X_train, _, y_train, _ = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)

    for ratio in ratios_kN:
        k = max(1, int(len(X_train) * ratio))
        if k >= len(X_train):
            continue

        mi = mutual_info_regression(X_train, y_train, random_state=42, n_neighbors=k)
        mi_df = pd.DataFrame({'Feature': X_train.columns, 'MI': mi})
        mi_df = mi_df.sort_values(by='MI', ascending=False).reset_index(drop=True)
        top_features = mi_df['Feature'].tolist()[:top_k]

        jaccard_data[(p_samp, ratio)] = set(top_features)

# === Calcul des distances de Jaccard par rapport à la référence
ref_key = (100, ref_ratio)
if ref_key not in jaccard_data:
    raise ValueError("Référence absente : 100% de données, ratio k/N=0.16")

ref_features = jaccard_data[ref_key]

heatmap_matrix = []

for ratio in ratios_kN:
    row = []
    for p_samp in percent_samples:
        key = (p_samp, ratio)
        ref_key = (p_samp, ref_ratio)  # référence dynamique ici !

        if key not in jaccard_data or ref_key not in jaccard_data:
            row.append(np.nan)
        else:
            current_set = jaccard_data[key]
            ref_features = jaccard_data[ref_key]

            intersection = len(current_set & ref_features)
            union = len(current_set | ref_features)
            jaccard_index = intersection / union if union > 0 else 0
            jaccard_distance = 1 - jaccard_index
            row.append(jaccard_distance)
    heatmap_matrix.append(row)


# === Affichage de la heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap="viridis",
            xticklabels=percent_samples, yticklabels=ratios_kN,
            annot_kws={"weight": "bold"})
plt.xlabel("Taille de l'échantillon (%)", fontweight='bold')
plt.ylabel("k / N", fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.savefig(f"../../Figures/Problèmes identifiés/wing_{col}_Jaccard.pdf", dpi=500)
plt.show()


# Quadrillation des meilleurs hyper-paramètres pour Random Forest

# === Grille d'hyperparamètres ===

param_grid = {
    "n_estimators": [200],
    "max_depth": [ 20],
    "min_samples_split": [2],
    "min_samples_leaf": [4],
}

grid = list(ParameterGrid(param_grid))

# === Cas à traiter ===

cas_array = ["wing"]

for cas in cas_array:
    entrees = choix(cas)[0]
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)
    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    for i in tqdm(target.columns, desc=f"Traitement de {cas} sorite"):
        # i = target.columns[0]
        X_train, X_test, y_train, y_test = train_test_split(data, target[i], test_size=0.2, random_state=42)

        best_r2 = -np.inf
        best_model = None
        best_params = None

        for params in grid:
            rf = RandomForestRegressor(random_state=42, **params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            score = r2_score(y_test, y_pred)

            if score > best_r2:
                best_r2 = score
                best_model = rf
                best_params = params
                best_pred = y_pred
                best_importance = rf.feature_importances_

        print(f"\n--- {cas.upper()} | Target: {i} ---")
        print(f"Meilleurs hyperparamètres : {best_params}")
        print(f"Meilleur R² sur test : {best_r2:.4f}")

        save_prefix = f"../../Figures/Problèmes identifiés/RF/{cas}_{i}_optimal_"
        with open(f"{save_prefix}params.txt", "w") as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")

        plot_importance(best_importance, X_train.columns, save_path=f"{save_prefix}rf1.pdf")
        plot_prediction_scatter(y_test.values, best_pred, save_path=f"{save_prefix}rf2.pdf")

        # Top 20% des variables
        top_idx = np.argsort(best_importance)[-(math.ceil(len(X_train.columns)/5)):]
        top_features = X_train.columns[top_idx]

        rf_top = RandomForestRegressor(random_state=42, **best_params)
        rf_top.fit(X_train[top_features], y_train)
        y_pred_top = rf_top.predict(X_test[top_features])
        plot_prediction_scatter(y_test.values, y_pred_top, save_path=f"{save_prefix}rf3.pdf")

# Random forest mauvaise prédiction mais bonne importance ? 

def load_best_params(filepath):
    params = {}
    with open(filepath, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            # Convertir les valeurs au bon type
            if value == "None":
                value = None
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            params[key] = value
    return params

cas_array = ["wing"]

for cas in cas_array:
    entrees = choix(cas)[0]
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)
    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    for i in tqdm(target.columns, desc=f"Traitement de {cas}"):

        X_train, X_test, y_train, y_test = train_test_split(data, target[i], test_size=0.2, random_state=42)

        save_prefix = f"../../Figures/Problèmes identifiés/RF/{cas}_{i}_optimal_"
        params_path = f"{save_prefix}params.txt"
        best_params = load_best_params(params_path)
        best_params["bootstrap"] = True  # Activation explicite du bootstrap si nécessaire

        # === Random Forest avec meilleurs paramètres ===
        rf = RandomForestRegressor(random_state=42, **best_params)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        importances = rf.feature_importances_

        # === Top 20% des variables ===
        top_k = math.ceil(len(X_train.columns) / 5)
        top_idx = np.argsort(importances)[-top_k:]
        top_features = X_train.columns[top_idx]

        # === Régression linéaire sur top 20% ===
        lin_reg = LinearRegression()
        lin_reg.fit(X_train[top_features], y_train)
        y_pred_lin = lin_reg.predict(X_test[top_features])

        # === Plot ===
        plot_prediction_scatter(y_test.values, y_pred_lin, save_path=f"{save_prefix}linreg_top20.pdf")