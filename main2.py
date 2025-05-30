from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
import winsound
from sklearn.metrics import r2_score, mean_absolute_error

cas = "wing"

entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

# Division en __% entraînement et __% test
X_train, X_test, y_train, y_test = train_test_split(data, target["3"], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

rf = RandomForestRegressor(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [10, 20, 50, 100, 200],
    'min_samples_leaf': [5, 10, 20, 50],
    'max_features': ['auto', 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=3,
    verbose=1,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)

random_search.fit(X_train, y_train)

print("\n✅ Meilleurs paramètres trouvés :")
print(random_search.best_params_)

best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("\n📈 Performance sur test set :")
print(f"RMSE : {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R² : {r2_score(y_test, y_pred):.4f}")

winsound.Beep(1000, 3000)  

# {'n_estimators': 300, 'min_samples_split': 50, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'max_depth': 20}



# Information mutuelle
mi_df =  mutual_info_regression(X_train, y_train, random_state=model.random_state, n_neighbors=348)


# Plot quantiles
# Trier les valeurs d'information mutuelle par ordre décroissant
mi_df = pd.DataFrame({'Feature': X_train.columns, 'Mutual Information': mi_df})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Calcul des quantiles
q1, q2, q3, q4 = np.percentile(mi_df['Mutual Information'], [25, 50, 75, 100])

# Affichage des quantiles
print(f"\nQuantiles de l'information mutuelle :")
print(f"1er quartile (Q1 - 25%) : {q1:.6f}")
print(f"2e quartile (Q2 - médiane - 50%) : {q2:.6f}")
print(f"3e quartile (Q3 - 75%) : {q3:.6f}")
print(f"4e quartile (Q4 - max - 100%) : {q4:.6f}")

# Graphique 1 : Information mutuelle triée par ordre décroissant
plt.figure(figsize=(10, 6))
plt.barh(mi_df['Feature'], mi_df['Mutual Information'], color='royalblue')
plt.xlabel("Information mutuelle")
plt.ylabel("Features")
plt.title("Information mutuelle par sortie (tri décroissant)")
plt.gca().invert_yaxis()
plt.yticks([])
plt.show()

epsilon = 1e-8
nb_presque_zero = (mi_df['Mutual Information'] < epsilon).sum()
print(f"Nombre de features avec une information mutuelle quasi nulle (< {epsilon}) : {nb_presque_zero}")




# Algo sélection de variables
evaluation_params = ['MI','Specific',["0"]]
stopping_params = ['subset_max', 460, 'any', evaluation_params[1], evaluation_params[2]]
MI_arg = 'LinearRegression' #ou 'RandomForest'

df = subset_generation(model, evaluation_params, stopping_params, data, target, data_full, target_full, MI_arg)



# Pour enregistrer les groupes trouvés
# Sélection des indices des features appartenant au cluster 0
indices_cluster = mi_df[mi_df['Mutual Information'] > epsilon].index
indices_cluster = mi_df[mi_df['cluster'] == 0].index
print(len(indices_cluster))
subset =[]
# Sauvegarde dans un fichier .txt
with open("indices_cluster.txt", "w") as f:
    f.write("[")
    for index in indices_cluster:
        f.write(",")
        f.write(str(index))
        subset.append(str(index))
    f.write("]")



model_all = LinearRegression()
model_all.fit(X_train[subset], y_train)
y = model_all.predict(X_test[subset])
print("R² Linear Regression (toutes les features) :", r2_score(y_test, y))
print("Mean Absolute Error (LR) :", mean_absolute_error(y_test, y))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm

# Modèle de base avec toutes les features
model_all = LinearRegression()
model_all.fit(X_train, y_train)
y_pred_all = model_all.predict(X_test)
r2_full = r2_score(y_test, y_pred_all)

# Paramètres
n_neighbors_list = [3, 15, 50, 100, 300, 696, 1392]
percentiles = [100, 95, 90, 80, 50, 30, 10]
labels = [f"{p}%" for p in percentiles] + ["> ε"]  # Ajout de la ligne spéciale

# Nouvelle matrice avec une ligne de plus
heatmap_matrix = np.zeros((len(percentiles) + 1, len(n_neighbors_list)))

# Boucle principale
for j, k in enumerate(tqdm(n_neighbors_list, desc="Calculs")):
    # Calcul de l'information mutuelle
    mi = mutual_info_regression(X_train, y_train, random_state=42, n_neighbors=k)
    mi_df = pd.DataFrame({'Feature': X_train.columns, 'Mutual Information': mi})
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

    # Boucle sur les percentiles
    for i, p in enumerate(percentiles):
        if p == 100:
            subset = mi_df['Feature'].tolist()
        else:
            n = int(len(mi_df) * (p / 100))
            subset = mi_df.iloc[:n]['Feature'].tolist()

        if len(subset) > 0:
            model = LinearRegression()
            model.fit(X_train[subset], y_train)
            y_pred = model.predict(X_test[subset])
            r2_sub = r2_score(y_test, y_pred)
            perte_r2 = (1 - (r2_sub / r2_full)) * 100
        else:
            perte_r2 = 100

        heatmap_matrix[i, j] = perte_r2

    # Ligne spéciale : IM > epsilon
    epsilon = 1e-8
    subset_epsilon = mi_df[mi_df['Mutual Information'] > epsilon]['Feature'].tolist()

    if len(subset_epsilon) > 0:
        model = LinearRegression()
        model.fit(X_train[subset_epsilon], y_train)
        y_pred = model.predict(X_test[subset_epsilon])
        r2_sub = r2_score(y_test, y_pred)
        perte_r2 = (1 - (r2_sub / r2_full)) * 100
    else:
        perte_r2 = 100

    heatmap_matrix[-1, j] = perte_r2  # dernière ligne

# DataFrame pour la heatmap
df_heatmap = pd.DataFrame(heatmap_matrix, 
                          index=labels, 
                          columns=n_neighbors_list)

# Affichage
plt.figure(figsize=(12, 8))
sns.heatmap(df_heatmap, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': 'Perte R² (%)'})
plt.title("Perte relative de performance (R²) selon % de features sélectionnées et n_neighbors")
plt.xlabel("n_neighbors pour mutual_info")
plt.ylabel("Top % de features ou seuil IM > ε")
plt.tight_layout()
plt.show()

winsound.Beep(1000, 3000)  


# Info mutuelle avec neighbor features différentes ?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
pos0=["0","1","2","3"]

for i in pos0:
    # Chargement des données
    cas = "dome"

    entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
    sorties = choix(cas)[1]

    model = K_structuraux(entrees, sorties)

    data_full = model.donnees(cas)
    target_full = model.sortie(cas)

    data = data_full[[str(i) for i in entrees]]
    target = target_full[[str(i) for i in sorties]]

    # Division en __% entraînement et __% test
    X_train, X_test, y_train, y_test = train_test_split(data, target[i], test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



    # Liste des valeurs de n_neighbors à tester
    n_neighbors_list = [3, 5, 10, 20, 30, 50, 80, 100, 300, 696, 1000, 1300]

    # Taille du top % à comparer
    top_percent = 20
    top_ratio = top_percent / 100
    top_n = int(X_train.shape[1] * top_ratio)

    # Fonction pour obtenir les top features selon IM
    def top_features(X, y, n_neighbors, top_n):
        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=n_neighbors)
        mi_df = pd.Series(mi, index=X.columns)
        return set(mi_df.sort_values(ascending=False).head(top_n).index)

    # Top features de référence (avec n_neighbors = 3)
    reference_top = top_features(X_train, y_train, n_neighbors=3, top_n=top_n)

    # Calcul du pourcentage d'intersection avec le top de référence
    overlap_percentages = []
    for k in tqdm(n_neighbors_list, desc="Comparaison"):
        current_top = top_features(X_train, y_train, n_neighbors=k, top_n=top_n)
        intersection = reference_top.intersection(current_top)
        percentage_overlap = (len(intersection) / top_n) * 100
        overlap_percentages.append(percentage_overlap)

    # Tracé du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(n_neighbors_list, overlap_percentages, marker='o', linestyle='-', color='teal')
    plt.title(f"Stabilité du Top {top_percent}% des features selon n_neighbors")
    plt.xlabel("n_neighbors")
    plt.ylabel(f"% de features en commun avec n_neighbors=3")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.tight_layout()
    plt.show()





# ACP exemple

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Chargement des données
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

# Standardisation des données
X_std = StandardScaler().fit_transform(X)

# === Visualisation AVANT ACP ===
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_std[y == i, 0], X_std[y == i, 1], label=target_name, alpha=0.7)

plt.xlabel(f'{feature_names[0]} (standardisé)')
plt.ylabel(f'{feature_names[1]} (standardisé)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Application de l'ACP ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)

# === Visualisation APRÈS ACP ===
plt.figure(figsize=(8, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name, alpha=0.7)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Information mutuelle 4D clusters

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

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    print("⚠️ HDBSCAN n'est pas installé. Utilise `pip install hdbscan` pour l'utiliser.")
    HDBSCAN_AVAILABLE = False

# Ton modèle et tes données
cas = "wing"
entrees = choix(cas)[0]
sorties = choix(cas)[1]
model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

X_train, X_test, y_train, y_test = train_test_split(data, target[["0", "1", "2","3"]], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

mi_df = mutual_info(model, X_train, y_train, 1)

coeff_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coeff_0': mi_df[:,0],
    'Coeff_1': mi_df[:,1],
    'Coeff_2': mi_df[:,2],
    'Coeff_3': mi_df[:,3]
})

# Normalisation
scaler = StandardScaler()
coeff_scaled = scaler.fit_transform(coeff_df[['Coeff_0', 'Coeff_1', 'Coeff_2', 'Coeff_3']])

# Application DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=2)
coeff_df['Cluster_DBSCAN'] = dbscan.fit_predict(coeff_scaled)

# Application HDBSCAN si dispo
if HDBSCAN_AVAILABLE:
    hdb = hdbscan.HDBSCAN(min_cluster_size=2)
    coeff_df['Cluster_HDBSCAN'] = hdb.fit_predict(coeff_scaled)

# Réduction dimensionnelle (t-SNE ou PCA)
pca = PCA(n_components=2)
reduced_2d = pca.fit_transform(coeff_scaled)
coeff_df['Dim1'] = reduced_2d[:, 0]
coeff_df['Dim2'] = reduced_2d[:, 1]

# Affichage 2D des clusters DBSCAN
plt.figure(figsize=(8, 6))
sns.scatterplot(data=coeff_df, x='Dim1', y='Dim2', hue='Cluster_DBSCAN', palette='tab10', s=100)
plt.title("Clusters DBSCAN (après réduction en 2D)")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Affichage HDBSCAN si disponible
if HDBSCAN_AVAILABLE:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=coeff_df, x='Dim1', y='Dim2', hue='Cluster_HDBSCAN', palette='tab10', s=100)
    plt.title("Clusters HDBSCAN (après réduction en 2D)")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Affichage des résultats texte
print("Clusters DBSCAN détectés :\n", coeff_df[['Feature', 'Cluster_DBSCAN']].sort_values(by='Cluster_DBSCAN'))

if HDBSCAN_AVAILABLE:
    print("\nClusters HDBSCAN détectés :\n", coeff_df[['Feature', 'Cluster_HDBSCAN']].sort_values(by='Cluster_HDBSCAN'))


## Sans outliers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def clusterwise_feature_selection_multioutput(coeff_df, X_train, X_test, y_train, y_test, max_features_per_cluster=3):
    if "Cluster_HDBSCAN" not in coeff_df.columns or not HDBSCAN_AVAILABLE:
        print("⚠️ HDBSCAN non disponible ou non appliqué.")
        return

    coeff_df = coeff_df.copy()
    clusters = coeff_df[coeff_df["Cluster_HDBSCAN"] != -1]["Cluster_HDBSCAN"].unique()
    
    r2_results = {f"Sortie {i}": [] for i in range(4)}  # Pour stocker les R²
    nb_features_list = []

    for k in range(1, max_features_per_cluster + 1):
        selected_features = []

        for cluster in clusters:
            cluster_data = coeff_df[coeff_df["Cluster_HDBSCAN"] == cluster].copy()
            cluster_data["mean_mi"] = cluster_data[[f"Coeff_{i}" for i in range(4)]].mean(axis=1)
            best_features = cluster_data.sort_values(by="mean_mi", ascending=False).head(k)["Feature"].tolist()
            selected_features.extend(best_features)

        selected_features = list(set(selected_features))  # Unicité
        nb_features_list.append(len(selected_features))

        # Régression linéaire pour chaque sortie séparément
        for i in range(4):
            lr = LinearRegression()
            lr.fit(X_train[selected_features], y_train[str(i)])
            y_pred = lr.predict(X_test[selected_features])
            r2 = r2_score(y_test[str(i)], y_pred)
            r2_results[f"Sortie {i}"].append(r2)

            print(f"[Sortie {i}] {len(selected_features)} features (k={k}/cluster) ➜ R² = {r2:.4f}")

    return nb_features_list, r2_results

# ➕ Appel de la fonction
nb_features_list, r2_results = clusterwise_feature_selection_multioutput(
    coeff_df, X_train, X_test, y_train, y_test, max_features_per_cluster=200
)

# 🎯 Calcul des R² de référence (toutes les features) pour chaque sortie
r2_refs = []
for i in range(4):
    lr_full = LinearRegression()
    lr_full.fit(X_train, y_train[str(i)])
    y_pred_full = lr_full.predict(X_test)
    r2_full = r2_score(y_test[str(i)], y_pred_full)
    r2_refs.append(r2_full)
    print(f"🎯 R² contrôle (sortie {i}, toutes features) = {r2_full:.4f}")

# 📈 Tracé du graphique
plt.figure(figsize=(10, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
linestyles = ['--', '-', '-.', ':']

for i in range(4):
    sortie = f"Sortie {i}"
    plt.plot(nb_features_list, r2_results[sortie], marker='o',
             linestyle=linestyles[i], label=sortie, color=colors[i])
    
    # 🔹 Ligne horizontale de référence
    plt.axhline(y=r2_refs[i], color=colors[i], linestyle='dotted',
                label=f"Réf sortie {i} (toutes features)")
    plt.text(nb_features_list[-1], r2_refs[i] + 0.005,
             f"R²={r2_refs[i]:.3f}", color=colors[i])

plt.xlabel("Nombre total de features sélectionnées")
plt.ylabel("Score R²")
plt.title("Évolution du R² pour chaque sortie (cluster-wise selection)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# Avec outliers


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def clusterwise_feature_selection_multioutput(coeff_df, X_train, X_test, y_train, y_test, max_features_per_cluster=3):
    if "Cluster_HDBSCAN" not in coeff_df.columns or not HDBSCAN_AVAILABLE:
        print("⚠️ HDBSCAN non disponible ou non appliqué.")
        return

    coeff_df = coeff_df.copy()
    clusters = coeff_df["Cluster_HDBSCAN"].unique()  # Inclut les outliers (-1)
    r2_results = {f"Sortie {i}": [] for i in range(4)}
    nb_features_list = []

    for k in range(1, max_features_per_cluster + 1):
        selected_features = []

        for cluster in clusters:
            cluster_data = coeff_df[coeff_df["Cluster_HDBSCAN"] == cluster].copy()
            cluster_data["mean_mi"] = cluster_data[[f"Coeff_{i}" for i in range(4)]].mean(axis=1)
            best_features = cluster_data.sort_values(by="mean_mi", ascending=False).head(k)["Feature"].tolist()
            selected_features.extend(best_features)

        selected_features = list(set(selected_features))  # Unicité
        nb_features_list.append(len(selected_features))

        for i in range(4):
            lr = LinearRegression()
            lr.fit(X_train[selected_features], y_train[str(i)])
            y_pred = lr.predict(X_test[selected_features])
            r2 = r2_score(y_test[str(i)], y_pred)
            r2_results[f"Sortie {i}"].append(r2)

            print(f"[Sortie {i}] {len(selected_features)} features (k={k}/cluster) ➜ R² = {r2:.4f}")

    return nb_features_list, r2_results

# ➕ Appel de la fonction mise à jour
nb_features_list, r2_results = clusterwise_feature_selection_multioutput(
    coeff_df, X_train, X_test, y_train, y_test, max_features_per_cluster=200
)

# 🎯 Calcul des R² de référence (toutes les features) pour chaque sortie
r2_refs = []
for i in range(4):
    lr_full = LinearRegression()
    lr_full.fit(X_train, y_train[str(i)])
    y_pred_full = lr_full.predict(X_test)
    r2_full = r2_score(y_test[str(i)], y_pred_full)
    r2_refs.append(r2_full)
    print(f"🎯 R² contrôle (sortie {i}, toutes features) = {r2_full:.4f}")

# 📈 Tracé du graphique
plt.figure(figsize=(10, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
linestyles = ['--', '-', '-.', ':']

for i in range(4):
    sortie = f"Sortie {i}"
    plt.plot(nb_features_list, r2_results[sortie], marker='o',
             linestyle=linestyles[i], label=sortie, color=colors[i])
    
    plt.axhline(y=r2_refs[i], color=colors[i], linestyle='dotted',
                label=f"Réf sortie {i} (toutes features)")
    plt.text(nb_features_list[-1], r2_refs[i] + 0.005,
             f"R²={r2_refs[i]:.3f}", color=colors[i])

plt.xlabel("Nombre total de features sélectionnées")
plt.ylabel("Score R²")
plt.title("Évolution du R² pour chaque sortie (cluster-wise + outliers)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Avec seulement les outliers
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def feature_selection_outliers_only(coeff_df, X_train, X_test, y_train, y_test, max_features_per_outlier=3):
    if "Cluster_HDBSCAN" not in coeff_df.columns or not HDBSCAN_AVAILABLE:
        print("⚠️ HDBSCAN non disponible ou non appliqué.")
        return

    coeff_df = coeff_df.copy()
    
    # Sélection des features qui sont dans les outliers uniquement (Cluster_HDBSCAN == -1)
    outlier_data = coeff_df[coeff_df["Cluster_HDBSCAN"] == -1].copy()
    r2_results = {f"Sortie {i}": [] for i in range(4)}
    nb_features_list = []

    for k in range(1, max_features_per_outlier + 1):
        selected_features = []

        # On sélectionne les k meilleures features dans les outliers, basées sur la moyenne des informations mutuelles
        outlier_data["mean_mi"] = outlier_data[[f"Coeff_{i}" for i in range(4)]].mean(axis=1)
        best_features = outlier_data.sort_values(by="mean_mi", ascending=False).head(k)["Feature"].tolist()
        selected_features.extend(best_features)

        selected_features = list(set(selected_features))  # Unicité
        nb_features_list.append(len(selected_features))

        for i in range(4):
            lr = LinearRegression()
            lr.fit(X_train[selected_features], y_train[str(i)])
            y_pred = lr.predict(X_test[selected_features])
            r2 = r2_score(y_test[str(i)], y_pred)
            r2_results[f"Sortie {i}"].append(r2)

            print(f"[Sortie {i}] {len(selected_features)} features (k={k}/outliers) ➜ R² = {r2:.4f}")

    return nb_features_list, r2_results

# ➕ Lancer la fonction mise à jour
nb_features_list, r2_results = feature_selection_outliers_only(
    coeff_df, X_train, X_test, y_train, y_test, max_features_per_outlier=69
)

# 🎯 Calcul des R² de référence (toutes les features) pour chaque sortie
r2_refs = []
for i in range(4):
    lr_full = LinearRegression()
    lr_full.fit(X_train, y_train[str(i)])
    y_pred_full = lr_full.predict(X_test)
    r2_full = r2_score(y_test[str(i)], y_pred_full)
    r2_refs.append(r2_full)
    print(f"🎯 R² contrôle (sortie {i}, toutes features) = {r2_full:.4f}")

# 📈 Tracé du graphique
plt.figure(figsize=(10, 6))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
linestyles = ['--', '-', '-.', ':']

for i in range(4):
    sortie = f"Sortie {i}"
    plt.plot(nb_features_list, r2_results[sortie], marker='o',
             linestyle=linestyles[i], label=sortie, color=colors[i])
    
    plt.axhline(y=r2_refs[i], color=colors[i], linestyle='dotted',
                label=f"Réf sortie {i} (toutes features)")
    plt.text(nb_features_list[-1], r2_refs[i] + 0.005,
             f"R²={r2_refs[i]:.3f}", color=colors[i])

plt.xlabel("Nombre total de features sélectionnées parmi les outliers")
plt.ylabel("Score R²")
plt.title("Évolution du R² pour chaque sortie (seulement les outliers)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Régression linéaire par groupe de variables sans outliers 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Étape 1 : sélectionner uniquement les clusters valides (sans les -1)
valid_clusters = coeff_df['Cluster_HDBSCAN'].unique()
valid_clusters = [c for c in valid_clusters if c != -1]

# Étape 2 : créer le dictionnaire de features par cluster (sans outliers)
cluster_features = {
    c: coeff_df[coeff_df['Cluster_HDBSCAN'] == c]['Feature'].tolist()
    for c in valid_clusters
}

# Étape 3 : transformer les données : une variable = moyenne des features du cluster
def transform_clustered_data(X, clusters):
    X_new = pd.DataFrame(index=X.index)
    for cluster_id, feature_list in clusters.items():
        X_new[str(cluster_id)] = X[feature_list].mean(axis=1)
    return X_new

X_train_clustered = transform_clustered_data(X_train, cluster_features)
X_test_clustered = transform_clustered_data(X_test, cluster_features)

# Étape 4 : entraîner des modèles de régression linéaire pour chaque sortie
r2_scores = {}
for i in range(4):
    reg = LinearRegression()
    reg.fit(X_train_clustered, y_train[str(i)])
    y_pred = reg.predict(X_test_clustered)
    r2 = r2_score(y_test[str(i)], y_pred)
    r2_scores[f"Sortie {i}"] = r2

# Étape 5 : affichage
print("\n📊 Scores R² par sortie (outliers exclus) :")
for sortie, score in r2_scores.items():
    print(f"{sortie} : {score:.4f}")


# Régression linéaire par groupe de variables avec outliers comme classe à part

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Étape 1 : récupérer les clusters valides (exclure les outliers -1)
valid_clusters = coeff_df['Cluster_HDBSCAN'].unique()
valid_clusters = [c for c in valid_clusters if c != -1]

# Étape 2 : créer le dictionnaire de features par cluster
cluster_features = {
    c: coeff_df[coeff_df['Cluster_HDBSCAN'] == c]['Feature'].tolist()
    for c in valid_clusters
}

# Étape 3 : ajouter tous les outliers dans un seul cluster "outliers"
outliers = coeff_df[coeff_df['Cluster_HDBSCAN'] == -1]['Feature'].tolist()
if outliers:  # S'il y a bien des outliers
    cluster_features["outliers"] = outliers

# Étape 4 : transformer les données en moyennes par cluster
def transform_clustered_data(X, clusters):
    X_new = pd.DataFrame(index=X.index)
    for cluster_id, feature_list in clusters.items():
        X_new[str(cluster_id)] = X[feature_list].mean(axis=1)
    return X_new

X_train_clustered = transform_clustered_data(X_train, cluster_features)
X_test_clustered = transform_clustered_data(X_test, cluster_features)

# Étape 5 : entraînement de modèles de régression linéaire sur chaque sortie
r2_scores = {}
for i in range(4):
    reg = LinearRegression()
    reg.fit(X_train_clustered, y_train[str(i)])
    y_pred = reg.predict(X_test_clustered)
    r2 = r2_score(y_test[str(i)], y_pred)
    r2_scores[f"Sortie {i}"] = r2

# Étape 6 : affichage des résultats
print("\n📊 Scores R² par sortie avec moyennes de clusters (outliers regroupés) :")
for sortie, score in r2_scores.items():
    print(f"{sortie} : {score:.4f}")




# Sélection intelligente des clusters #1 et #2 partagent bcp d'information mutuelle

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

# === Données ===
cas = "wing"
entrees = choix(cas)[0]
sorties = choix(cas)[1]
model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

X_train, X_test, y_train, y_test = train_test_split(data, target[["0", "1", "2", "3"]], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Calcul de l'information mutuelle avec n_neighbors=169 pour 0, 1 et 3
mi_df0 = mutual_info_regression(X_train, y_train["0"], n_neighbors=169, random_state=model.random_state)
mi_df1 = mutual_info_regression(X_train, y_train["1"], n_neighbors=169, random_state=model.random_state)
mi_df3 = mutual_info_regression(X_train, y_train["3"], n_neighbors=169, random_state=model.random_state)

coeff_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coeff_0': mi_df0,
    'Coeff_1': mi_df1,
    'Coeff_3': mi_df3
})

# === Clustering HDBSCAN sur l'espace 3D (Coeff_0, Coeff_1, Coeff_3) ===
mi_subset = coeff_df[['Coeff_0', 'Coeff_1', 'Coeff_3']]

scaler = StandardScaler()
mi_scaled = scaler.fit_transform(mi_subset)

clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
clusters = clusterer.fit_predict(mi_scaled)
coeff_df['Cluster_HDBSCAN_0_1_3'] = clusters

# === Affichage 3D des clusters ===
fig = px.scatter_3d(
    coeff_df,
    x='Coeff_0',
    y='Coeff_1',
    z='Coeff_3',
    color=coeff_df['Cluster_HDBSCAN_0_1_3'].astype(str),
    hover_name='Feature',
    title="Clustering HDBSCAN sur (Coeff_0, Coeff_1, Coeff_3)"
)
fig.show()

# === Régression linéaire groupée (HDBSCAN avec outliers inclus) ===

# Regroupement des features par clusters HDBSCAN, y compris les outliers (-1)
cluster_labels = coeff_df['Cluster_HDBSCAN_0_1_3'].unique()
cluster_labels.sort()

groups_dict = {}
for label in cluster_labels:
    feats = coeff_df[coeff_df['Cluster_HDBSCAN_0_1_3'] == label]['Feature'].tolist()
    group_name = f'hdb_outliers' if label == -1 else f'hdb_cluster_{label}'
    groups_dict[group_name] = feats

# Compression des features
def compress_features(X, groups_dict):
    compressed_data = {}
    for i, (group_name, features) in enumerate(groups_dict.items()):
        valid_feats = [f for f in features if f in X.columns]
        compressed_data[f'group_{i}'] = X[valid_feats].mean(axis=1)
    return pd.DataFrame(compressed_data)

X_train_grouped = compress_features(X_train, groups_dict)
X_test_grouped = compress_features(X_test, groups_dict)

# Régression linéaire et affichage des scores R² pour les sorties 0, 1, 3
r2_scores = {}

for col in y_train.columns:
    if col not in ['0', '1', '3']:
        continue
    reg = LinearRegression()
    reg.fit(X_train_grouped, y_train[col])
    y_pred = reg.predict(X_test_grouped)
    r2 = r2_score(y_test[col], y_pred)
    r2_scores[col] = r2
    print(f"✅ R² pour la sortie {col} : {r2:.4f}")





# Tests mutual+info+regression


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors

# === Données ===
cas = "dome"
entrees = choix(cas)[0]
sorties = choix(cas)[1]
model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

X_train, X_test, y_train, y_test = train_test_split(data, target[["0"]], test_size=0.2, random_state=42) 
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialisation des variables pour stocker les résultats
neighbor_counts = [3,5,15,50,100,300,696,696*2]  # Nombre de voisins
sample_sizes = [0.1, 0.2, 0.3, 0.4, 0.5,0.8,0.9,0.99]  # Tailles des échantillons

# Stockage des résultats dans un DataFrame
results = []

# Calcul de l'information mutuelle pour différentes combinaisons
for neighbors in neighbor_counts:
    for sample_size in sample_sizes:
        # Création de sous-échantillons
        X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, test_size=1 - sample_size, random_state=42)
        
        # Calcul de l'information mutuelle avec un nombre de voisins donné
        knn = NearestNeighbors(n_neighbors=neighbors)
        knn.fit(X_train_sub)
        mutual_info_values = mutual_info_regression(X_train_sub, y_train_sub.values.ravel())

        # # Calcul de l'information mutuelle moyenne
        # avg_mutual_info = np.mean(mutual_info_values)

        # Définir un epsilon (seuil)
        epsilon = 1e-5

        # Compter le nombre de features avec une information mutuelle plus grande que epsilon
        count_above_epsilon = np.sum(mutual_info_values > epsilon)

        # Ajouter les résultats dans le DataFrame
        results.append([neighbors, sample_size, count_above_epsilon])


# Convertir les résultats en DataFrame
results_df = pd.DataFrame(results, columns=["Neighbors", "Sample Size", "Average Mutual Info"])


# Pivot pour Heatmap
heatmap_data = results_df.pivot(index="Neighbors", columns="Sample Size", values="Average Mutual Info")

# Affichage de la heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=0.5)
plt.title('Heatmap de nombre de zéros de l\'information mutuelle')
plt.xlabel('Taille de l\'échantillon')
plt.ylabel('Nombre de voisins')
plt.show()



# Blip blop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

# === Données ===
cas = "dome"
entrees = choix(cas)[0]
sorties = choix(cas)[1]
model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

X_train, X_test, y_train, y_test = train_test_split(data, target[["0"]], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

neighbor_counts = [3, 5, 10, 50, 100, 200, 500,650, 696, 700, 1000, 1300]

# Stockage des résultats
results = {}

# Calcul de l'information mutuelle pour chaque nombre de voisins
for neighbors in neighbor_counts:
    # Calcul de l'information mutuelle avec un nombre de voisins donné
    mutual_info_values = mutual_info_regression(X_train, y_train.values.ravel(), n_neighbors=neighbors)
    
    # Trier les valeurs d'information mutuelle par ordre croissant
    sorted_mutual_info = np.sort(mutual_info_values)
    
    # Stocker les résultats pour chaque nombre de voisins
    results[neighbors] = sorted_mutual_info

# Affichage des résultats pour chaque nombre de voisins
for neighbors, sorted_mutual_info in results.items():
    print(f"\nInformation mutuelle pour {neighbors} voisins:")
    for feature_idx, mutual_info in enumerate(sorted_mutual_info):
        print(f"Feature {feature_idx}: {mutual_info:.4f}")

# Affichage des résultats sous forme de graphique
plt.figure(figsize=(10, 8))
for neighbors, sorted_mutual_info in results.items():
    plt.plot(sorted_mutual_info, label=f'{neighbors} voisins')

plt.title('Information mutuelle par feature pour différents nombres de voisins')
plt.xlabel('Index des features')
plt.ylabel('Information mutuelle')
plt.legend(title='Nombre de voisins')
plt.grid(True)
plt.show()


winsound.Beep(1000, 2000)  


# Comparaison top20%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

# === Données ===
cas = "dome"
entrees = choix(cas)[0]
sorties = choix(cas)[1]
model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

X_train, X_test, y_train, y_test = train_test_split(data, target[["0"]], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Liste des nombres de voisins à tester
neighbor_counts = [3, 5, 10, 50, 100, 200, 500,650, 696, 700, 1000, 1300]
top_percent = 0.2  # top 20%

# Stockage des indices des meilleures features pour chaque nombre de voisins
top_features = {}

# Calcul de l'information mutuelle et sélection du top 20%
for neighbors in neighbor_counts:
    mi = mutual_info_regression(X_train, y_train.values.ravel(), n_neighbors=neighbors)
    top_k = int(len(mi) * top_percent)
    top_indices = np.argsort(mi)[-top_k:]
    top_features[neighbors] = set(top_indices)

# Référence : top 20% avec 169 voisins
ref_top = top_features[X_train.shape[1]]

# Comparaison avec la référence
overlap_percentages = {}
for neighbors, indices in top_features.items():
    common = ref_top.intersection(indices)
    overlap = 100 * len(common) / len(ref_top)
    overlap_percentages[neighbors] = overlap
    print(f"🔁 Chevauchement avec {neighbors} voisins : {overlap:.1f}%")

# Tracé du chevauchement
plt.figure(figsize=(8, 5))
x = [n for n in neighbor_counts ]
y = [overlap_percentages[n] for n in neighbor_counts ]

plt.plot(x, y, marker='o')
plt.axhline(100, color='gray', linestyle='--', label='Chevauchement parfait')
plt.title("Chevauchement (%) entre top 20% à 169 voisins et autres valeurs de voisins")
plt.xlabel("Nombre de voisins")
plt.ylabel("Chevauchement en % avec top 20% (169 voisins)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# Somme totale de l'information mutuelle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

# === Données ===
cas = "dome"
entrees = choix(cas)[0]
sorties = choix(cas)[1]
model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

X_train, X_test, y_train, y_test = train_test_split(data, target[["0"]], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Liste des nombres de voisins à tester
neighbor_counts = [3, 5, 10, 50, 100, 200, 500,650, 696, 700, 1000, 1300]

# Stockage des résultats (somme totale de l'information mutuelle)
total_mutual_info = {}

# Calcul de l'information mutuelle pour chaque nombre de voisins
for neighbors in neighbor_counts:
    # Calcul de l'information mutuelle avec un nombre de voisins donné
    mutual_info_values = mutual_info_regression(X_train, y_train.values.ravel(), n_neighbors=neighbors)
    
    # Somme totale de l'information mutuelle
    total_mutual_info[neighbors] = np.sum(mutual_info_values)

# Affichage des résultats sous forme de line plot
plt.figure(figsize=(10, 6))
plt.plot(list(total_mutual_info.keys()), list(total_mutual_info.values()), marker='o', color='skyblue', linestyle='-', linewidth=2)

plt.title('Somme totale de l\'information mutuelle pour différents nombres de voisins')
plt.xlabel('Nombre de voisins')
plt.ylabel('Somme totale de l\'information mutuelle')
plt.grid(True)
plt.tight_layout()
plt.show()


# Test masse Gaussienne

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, normaltest

# === Données ===
cas = "dome"
entrees = choix(cas)[0]
sorties = choix(cas)[1]
model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target[["0"]], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# === Affichage de l'histogramme et de la gaussienne ===
y = y_train.values.ravel()
mu, sigma = np.mean(y), np.std(y)

# Histogramme
plt.figure(figsize=(10, 6))
count, bins, ignored = plt.hist(y, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

# Courbe de densité gaussienne
x = np.linspace(min(y), max(y), 1000)
plt.plot(x, norm.pdf(x, mu, sigma), 'r--', linewidth=2, label=f'N({mu:.2f}, {sigma:.2f}²)')

plt.title("Histogramme de la masse avec densité gaussienne ajustée")
plt.xlabel("Valeurs de la masse")
plt.ylabel("Densité")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Tests de normalité ===
# Shapiro-Wilk Test
shapiro_stat, shapiro_p = shapiro(y)
print(f"Shapiro-Wilk test: statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.4e}")

# D'Agostino and Pearson's test
dagostino_stat, dagostino_p = normaltest(y)
print(f"D’Agostino and Pearson test: statistic = {dagostino_stat:.4f}, p-value = {dagostino_p:.4e}")

# === Interprétation simple ===
alpha = 0.05
if shapiro_p > alpha and dagostino_p > alpha:
    print("✅ Les deux tests suggèrent que y_train suit une distribution normale (au seuil de 5%).")
else:
    print("❌ Au moins un des tests rejette l'hypothèse de normalité pour y_train (au seuil de 5%).")



# Voir si le nombre de voisins est déterminé par le plan d'expérience

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau
from scipy.stats import kendalltau, spearmanr, pearsonr
from scipy.stats import wasserstein_distance
from sklearn.metrics import precision_score
import numpy as np

# === Données ===
cas = "wing"
entrees = choix(cas)[0]
sorties = choix(cas)[1]
model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in entrees]]
target = target_full[[str(i) for i in sorties]]

X_train, X_test, y_train, y_test = train_test_split(data, target[["0"]], test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# === Paramètres ===
n_neighbors_list = [3, 5, 15, 20, 50, 100,150, 162, 200,250, 300]  # Nombre de voisins
n_features = X_train.shape[1]
sample_sizes = [n_features * m for m in [ 3, 4,5, 8, 10, 15, 20, 25]]
n_repeats = 5  # moyenne sur plusieurs runs pour stabilité
top_k = max(1, int(0.2 * n_features))  # Top 20% des features

# === Stockage des résultats ===
results = {}  # {n_neighbors: {sample_size: {mean_total_im, std_total_im, mean_kendall_tau, mean_r2}}}

for n_neighbors in n_neighbors_list:
    results[n_neighbors] = {}
    for size in sample_sizes:
        total_ims = []
        rankings = []
        r2_scores = []
        vectors = []
        selected_topk = []

        for _ in range(n_repeats):
            idx = np.random.choice(X_train.index, size=min(size, len(X_train)), replace=False)
            X_sample = X_train.loc[idx]
            y_sample = y_train.loc[idx].values.ravel()

            im = mutual_info_regression(X_sample, y_sample, n_neighbors=n_neighbors, random_state=42)
            total_ims.append(np.sum(im))
            rankings.append(np.argsort(-im))
            vectors.append(im)
            selected_topk.append(set(np.argsort(-im)[:top_k]))

            # Sélection des top_k features pour la régression
            selected_features = [str(i) for i in np.argsort(-im)[:top_k]]
            model_lin = LinearRegression()
            model_lin.fit(X_sample[selected_features], y_sample)

            # Performance sur X_test
            r2 = model_lin.score(X_test[selected_features], y_test)
            r2_scores.append(r2)

        # Tau de Kendall moyen
        taus = []
        pearsons = []
        spearmans = []
        jaccards = []
        precisions = []
        wassers = []
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                tau, _ = kendalltau(rankings[i], rankings[j])
                taus.append(tau)
        mean_tau = np.mean(taus)
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                # Corrélation Pearson
                pearsons.append(pearsonr(vectors[i], vectors[j])[0])
                # Corrélation Spearman
                spearmans.append(spearmanr(vectors[i], vectors[j])[0])
                # Jaccard sur top_k
                jaccards.append(len(selected_topk[i] & selected_topk[j]) / len(selected_topk[i] | selected_topk[j]))
                # Precision@k
                inter = len(selected_topk[i] & selected_topk[j])
                precisions.append(inter / top_k)
                # Wasserstein distance
                wassers.append(wasserstein_distance(vectors[i], vectors[j]))

        results[n_neighbors][size] = {
            "mean_total_im": np.mean(total_ims),
            "std_total_im": np.std(total_ims),
            "mean_kendall_tau": mean_tau,
            "mean_r2": np.mean(r2_scores),
            "mean_pearson": np.mean(pearsons),
            "mean_spearman": np.mean(spearmans),
            "mean_jaccard": np.mean(jaccards),
            "mean_precision": np.mean(precisions),
            "mean_wasserstein": np.mean(wassers),
        }
        


# === Traces : Information mutuelle totale ===
plt.figure(figsize=(12, 6))
for n_neighbors in n_neighbors_list:
    plt.plot(sample_sizes,
             [results[n_neighbors][s]["mean_total_im"] for s in sample_sizes],
             marker='o',
             label=f"{n_neighbors} voisins")
plt.xlabel("Taille de l'échantillon")
plt.ylabel("Somme moyenne de l'information mutuelle")
plt.title("Information mutuelle moyenne vs. taille d'échantillon")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Heatmap : stabilité de classement (kendall tau) ===
tau_matrix = pd.DataFrame(index=n_neighbors_list, columns=sample_sizes)
for n in n_neighbors_list:
    for s in sample_sizes:
        tau_matrix.loc[n, s] = results[n][s]["mean_kendall_tau"]

plt.figure(figsize=(10, 6))
sns.heatmap(tau_matrix.astype(float), annot=True, cmap='viridis')
plt.title("Stabilité des classements (Kendall tau)")
plt.xlabel("Taille d'échantillon")
plt.ylabel("Nombre de voisins")
plt.tight_layout()
plt.show()

# === Heatmap : R² moyen ===
r2_matrix = pd.DataFrame(index=n_neighbors_list, columns=sample_sizes)
for n in n_neighbors_list:
    for s in sample_sizes:
        r2_matrix.loc[n, s] = results[n][s]["mean_r2"]

plt.figure(figsize=(10, 6))
sns.heatmap(r2_matrix.astype(float), annot=True, cmap='coolwarm')
plt.title("R² de la régression linéaire avec top 20% des features IM")
plt.xlabel("Taille d'échantillon")
plt.ylabel("Nombre de voisins")
plt.tight_layout()
plt.show()

def plot_heatmap(metric_name, cmap="mako"):
    matrix = pd.DataFrame(index=n_neighbors_list, columns=sample_sizes)
    for n in n_neighbors_list:
        for s in sample_sizes:
            matrix.loc[n, s] = results[n][s][metric_name]
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix.astype(float), annot=True, cmap=cmap)
    plt.title(f"Heatmap de la métrique : {metric_name}")
    plt.xlabel("Taille d'échantillon")
    plt.ylabel("Nombre de voisins")
    plt.tight_layout()
    plt.show()


plot_heatmap("mean_pearson", cmap="coolwarm")
plot_heatmap("mean_spearman", cmap="rocket")
plot_heatmap("mean_jaccard", cmap="Greens")
plot_heatmap("mean_precision", cmap="Blues")
plot_heatmap("mean_wasserstein", cmap="cividis")  # Plus bas = plus similaires


