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
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import hdbscan
from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour le plot 3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter


winsound.Beep(350, 2000)
winsound.Beep(1000, 2000)  


# === Fonction pour afficher et sauvegarder un scatter réel vs prédiction ===

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



def compress_features_joint(X, groups_dict):
    """
    Pour chaque groupe, calcule la somme (ou moyenne) des variables du groupe,
    mais conserve la forme originale pour que chaque variable puisse évoluer
    avec le même coefficient.
    
    Retourne :
        - X_grouped : DataFrame avec colonnes par groupe
        - group_matrix : DataFrame de même taille que X, où chaque colonne est 1 si elle appartient au groupe correspondant
    """
    X_grouped = pd.DataFrame(index=X.index)
    group_matrix = pd.DataFrame(0, index=X.columns, columns=groups_dict.keys())
    
    for group_name, features in groups_dict.items():
        valid = [f for f in features if f in X.columns]
        X_grouped[group_name] = X[valid].sum(axis=1)
        for f in valid:
            group_matrix.loc[f, group_name] = 1
            
    return X_grouped, group_matrix


def clustering(X_train, y_train, fit, min_cluster_size=2):
    coeffs = {}
    if fit == "RL":
        for col in y_train.columns:
            model = LinearRegression()
            model.fit(X_train, y_train[col])
            coeffs[f'Coeff_{col}'] = abs(model.coef_)
    elif fit == "RF":
        for col in tqdm(y_train.columns):
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train[col])
            coeffs[f'Coeff_{col}'] = model.feature_importances_
    elif fit == "IM":
        for col in tqdm(y_train.columns):
            model = mutual_info_regression(X_train, y_train[col], random_state=42, n_neighbors=math.ceil(len(X_train) * 0.16))
            coeffs[f'Coeff_{col}'] = model
    

    coeff_df = pd.DataFrame(coeffs)
    coeff_df['Feature'] = X_train.columns
    coeff_df = coeff_df.set_index('Feature')
    
    scaler = StandardScaler()
    coeff_scaled = scaler.fit_transform(coeff_df.values)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = clusterer.fit_predict(coeff_scaled)
    coeff_df['Cluster'] = clusters
    return coeff_df.reset_index()

# === Fonction pour sauvegarder les variables par cluster ===
def save_clusters(coeff_df, save_path):
    with open(save_path, 'w') as f:
        for cluster in sorted(coeff_df['Cluster'].unique()):
            features = coeff_df[coeff_df['Cluster'] == cluster]['Feature']
            f.write(" ".join(features) + '\n')


def plot_3d_clusters(coeff_df, save_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    # Colonnes à centrer-réduire
    features = ['Coeff_mass', 'Coeff_max.stress', 'Coeff_max.deflection']
    scaler = StandardScaler()
    coeff_scaled = scaler.fit_transform(coeff_df[features])

    # Récupération des clusters
    clusters = coeff_df['Cluster'].unique()
    n_clusters = len(clusters)

    # Palette de couleurs
    cmap = cm.get_cmap('tab20', n_clusters) if n_clusters <= 20 else cm.get_cmap('nipy_spectral', n_clusters)
    cluster_to_color = {cluster: cmap(i) for i, cluster in enumerate(clusters)}
    colors = coeff_df['Cluster'].map(cluster_to_color)

    # Affichage des points
    ax.scatter(
        coeff_scaled[:, 0],
        coeff_scaled[:, 1],
        coeff_scaled[:, 2],
        c=colors,
        s=50
    )

    # Format des ticks
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    for ticklabels in [ax.get_xticklabels(), ax.get_yticklabels(), ax.get_zticklabels()]:
        for label in ticklabels:
            label.set_fontweight('bold')

    # Labels
    ax.set_xlabel("Coeff_mass (centré réduit)", fontweight='bold')
    ax.set_ylabel("Coeff_max.stress (centré réduit)", fontweight='bold')
    ax.set_zlabel("Coeff_max.deflection (centré réduit)", fontweight='bold')
    ax.set_title(f"{n_clusters} clusters", fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=500)

    plt.show()


def group_features_by_cluster(coeff_df):
    groups_dict = {}
    for cluster in sorted(coeff_df['Cluster'].unique()):
        features = coeff_df[coeff_df['Cluster'] == cluster]['Feature'].tolist()
        group_name = f'cluster_{cluster}' if cluster != -1 else 'outliers'
        groups_dict[group_name] = features
    return groups_dict


# === Boucle principale === # Régression linéaire
cas_array = ["wing","dome","bldg"]

for cas in cas_array:
    entrees = choix(cas)[0]
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)
    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    coeff_df = clustering(X_train, y_train, "RL", min_cluster_size = 2)
    save_prefix = f"../../Figures/Groupement/RL/"
    save_clusters(coeff_df, os.path.join(save_prefix, f"{cas}_variables_par_cluster.txt"))
    plot_3d_clusters(coeff_df, os.path.join(save_prefix, f"{cas}_clusters_3D.pdf"))

    groups_dict = group_features_by_cluster(coeff_df)
    X_train_grouped, group_matrix = compress_features_joint(X_train, groups_dict)
    X_test_grouped, _ = compress_features_joint(X_test, groups_dict)


    for col in target.columns:
        reg = LinearRegression()
        reg.fit(X_train_grouped, y_train[col])
        
        # Prédiction au niveau des groupes
        y_pred_group = reg.predict(X_test_grouped)

        # Étendre la prédiction au niveau des variables originales :
        # Chaque variable reçoit le coefficient du groupe auquel elle appartient
        coef_group = pd.Series(reg.coef_, index=X_train_grouped.columns)
        
        # Étendre les coefficients aux variables initiales
        full_coefs = group_matrix.dot(coef_group)
        
        # Prédiction avec toutes les variables évoluant avec le même coef
        y_pred_full = X_test.dot(full_coefs) + reg.intercept_

        print(f"✅ R² pour la sortie {col} : {r2_score(y_test[col], y_pred_full):.4f}")
        
        plot_prediction_scatter(
            y_test[col].values, y_pred_full,
            save_path=os.path.join(save_prefix, f"{cas}_pred_scatter_y{col}.pdf")
        )


# === Boucle principale === # Ranfom Forest
cas_array = ["wing","dome","bldg"]

for cas in cas_array:
    entrees = choix(cas)[0]
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)
    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    coeff_df = clustering(X_train, y_train, "RF", min_cluster_size = 2)
    save_prefix = f"../../Figures/Groupement/RF/"
    save_clusters(coeff_df, os.path.join(save_prefix, f"{cas}_variables_par_cluster.txt"))
    plot_3d_clusters(coeff_df, os.path.join(save_prefix, f"{cas}_clusters_3D.pdf"))

    groups_dict = group_features_by_cluster(coeff_df)
    X_train_grouped, group_matrix = compress_features_joint(X_train, groups_dict)
    X_test_grouped, _ = compress_features_joint(X_test, groups_dict)

    for col in target.columns:
        reg = LinearRegression()
        reg.fit(X_train_grouped, y_train[col])
        y_pred = reg.predict(X_test_grouped)
        print(f"✅ R² pour la sortie {col} : {r2_score(y_test[col], y_pred):.4f}")
        plot_prediction_scatter(
    y_test[col].values, y_pred,
    save_path=os.path.join(save_prefix, f"{cas}_pred_scatter_y{col}.pdf")
)
        
# === Boucle principale === # Information Mutuelle
cas_array = ["wing","dome","bldg"]

for cas in cas_array:
    entrees = choix(cas)[0]
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)
    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in data_full.columns]]
    target = target_full[[str(i) for i in target_full.columns]]

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    coeff_df = clustering(X_train, y_train, "IM", min_cluster_size = 2)
    save_prefix = f"../../Figures/Groupement/IM/"
    save_clusters(coeff_df, os.path.join(save_prefix, f"{cas}_variables_par_cluster.txt"))
    plot_3d_clusters(coeff_df, os.path.join(save_prefix, f"{cas}_clusters_3D.pdf"))

    groups_dict = group_features_by_cluster(coeff_df)
    X_train_grouped, group_matrix = compress_features_joint(X_train, groups_dict)
    X_test_grouped, _ = compress_features_joint(X_test, groups_dict)

    for col in target.columns:
        reg = LinearRegression()
        reg.fit(X_train_grouped, y_train[col])
        y_pred = reg.predict(X_test_grouped)
        print(f"✅ R² pour la sortie {col} : {r2_score(y_test[col], y_pred):.4f}")
        plot_prediction_scatter(
    y_test[col].values, y_pred,
    save_path=os.path.join(save_prefix, f"{cas}_pred_scatter_y{col}.pdf")
)
