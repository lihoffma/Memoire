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
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from matplotlib import font_manager

winsound.Beep(1000, 2000)  

def plot_importance(importance_array, feature_names, save_path=None):
    """
    Affiche l'importance des variables à partir d'un tableau d'importances.
    
    Paramètres :
        - importance_array : array-like, importances (ex: |coef|, feature_importance)
        - feature_names : liste des noms des colonnes
        - save_path : chemin de sauvegarde éventuel
    """
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, importance_array, color='blue')
    plt.ylabel("Importance", fontweight='bold')
    plt.xticks(rotation=45, fontweight='bold')
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


# ACP
    # model: le cas test considéré
    # evaluation_params: les critères d'évaluation [(RandomForest, LinearRegression, PCA, MI, DynMI), (All, Specific, Mean), ([1,3])]
    # stopping_params: le critère d'arrêt [(subset_max, iter_max, R2, MSE), (5, 100, 0.9, 0.1), (any, all), (All, Specific, Mean), ([1,3])]
    # data_full: les données d'entrée complètes
    # target_full: les données de sortie complètes
# evaluation_params = ['PCA','All']
# stopping_params = ['iter_max', 100 , 'all', 'All']
# MI_arg = 0
# df = subset_generation(model, evaluation_params, stopping_params, data, target, data_full, target_full, MI_arg)

# Cas test diabétique
entrees = range(10)
model = K_Diabete(entrees)

data_full = model.donnees()
target_full = model.sortie()
data = data_full.copy()
target = target_full.copy()

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)


pca = PCA()
X_pca = pca.fit_transform(X_train)
importance = np.abs(pca.components_.T @ pca.explained_variance_ratio_)
explained_var = pca.explained_variance_ratio_

print("Variance expliquée par chaque composante principale :")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var:.4f}")


plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/ACP/diab_acp1.pdf")


# === Scatter plot des deux premières composantes avec la target en couleur ===
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train.values.ravel(), cmap='viridis', edgecolor='k')
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)", fontweight='bold')
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)", fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
# Colorbar
cb = plt.colorbar(scatter)
for t in cb.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("../../Figures/Jeux de données simples/ACP/diab_acp2.pdf", dpi=500)
plt.show()

# === Régression Linéaire avec toutes les variables ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# === Prédiction ===
y_pred = lr.predict(X_test)
plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/ACP/diab_acp3.pdf")

# === Sélection des deux variables les plus importantes ===
top2_idx = np.argsort(importance)[-2:]
top2_features = X_train.columns[top2_idx]

# === Filtrage des données sur ces deux variables ===
X_train_top2 = X_train[top2_features]
X_test_top2 = X_test[top2_features]

# === Régression Linéaire ===
lr = LinearRegression()
lr.fit(X_train_top2, y_train)

# === Prédiction ===
y_pred = lr.predict(X_test_top2)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/ACP/diab_acp4.pdf")


# Cas test mécanique Weight

entrees = list(range(9))
sorties = [0]
nombre = 1000

model = K_Mecanique(entrees, sorties, nombre)

data_full = model.donnees(nombre)
target_full = model.sortie(data_full)

# on prend les entrees et les sorties désirées de repectivement data et target en choississant les bonnes lignes
data = data_full[entrees]
target = target_full[sorties]
data = model.renommer(data,-1)
target = model.renommer(target,sorties[0])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

pca = PCA()
X_pca = pca.fit_transform(X_train)
importance = np.abs(pca.components_.T @ pca.explained_variance_ratio_)
explained_var = pca.explained_variance_ratio_

print("Variance expliquée par chaque composante principale :")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var:.4f}")

# Plot
plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/ACP/meca1_acp1.pdf")


# === Scatter plot des deux premières composantes avec la target en couleur ===
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train.values.ravel(), cmap='viridis', edgecolor='k')
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)", fontweight='bold')
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)", fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
# Colorbar
cb = plt.colorbar(scatter)
for t in cb.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("../../Figures/Jeux de données simples/ACP/meca1_acp2.pdf", dpi=500)
plt.show()

# === Régression Linéaire avec toutes les variables ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# === Prédiction ===
y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/ACP/meca1_acp3.pdf")



# === Sélection des deux variables les plus importantes ===
top2_idx = np.argsort(importance)[-2:]
top2_features = X_train.columns[top2_idx]

# === Filtrage des données sur ces deux variables ===
X_train_top2 = X_train[top2_features]
X_test_top2 = X_test[top2_features]

# === Régression Linéaire ===
lr = LinearRegression()
lr.fit(X_train_top2, y_train)

# === Prédiction ===
y_pred = lr.predict(X_test_top2)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/ACP/meca1_acp4.pdf")

# Cas test mécanique VC_middle

entrees = list(range(9))
sorties = [6]
nombre = 1000

model = K_Mecanique(entrees, sorties, nombre)

data_full = model.donnees(nombre)
target_full = model.sortie(data_full)

# on prend les entrees et les sorties désirées de repectivement data et target en choississant les bonnes lignes
data = data_full[entrees]
target = target_full[sorties]
data = model.renommer(data,-1)
target = model.renommer(target,sorties[0])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

pca = PCA()
X_pca = pca.fit_transform(X_train)
importance = np.abs(pca.components_.T @ pca.explained_variance_ratio_)
explained_var = pca.explained_variance_ratio_

print("Variance expliquée par chaque composante principale :")
for i, var in enumerate(explained_var):
    print(f"PC{i+1}: {var:.4f}")

# Plot
plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/ACP/meca2_acp1.pdf")


# === Scatter plot des deux premières composantes avec la target en couleur ===
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train.values.ravel(), cmap='viridis', edgecolor='k')
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}%)", fontweight='bold')
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}%)", fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
# Colorbar
cb = plt.colorbar(scatter)
for t in cb.ax.get_yticklabels():
    t.set_fontweight('bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("../../Figures/Jeux de données simples/ACP/meca2_acp2.pdf", dpi=500)
plt.show()

# === Régression Linéaire avec toutes les variables ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# === Prédiction ===
y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/ACP/meca2_acp3.pdf")



# === Sélection des deux variables les plus importantes ===
top2_idx = np.argsort(importance)[-2:]
top2_features = X_train.columns[top2_idx]

# === Filtrage des données sur ces deux variables ===
X_train_top2 = X_train[top2_features]
X_test_top2 = X_test[top2_features]

# === Régression Linéaire ===
lr = LinearRegression()
lr.fit(X_train_top2, y_train)

# === Prédiction ===
y_pred = lr.predict(X_test_top2)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/ACP/meca2_acp4.pdf")

# Régression linéaire

# Cas test diabétique

entrees = range(10)
model = K_Diabete(entrees)

data_full = model.donnees()
target_full = model.sortie()
data = data_full.copy()
target = target_full.copy()

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# === Régression Linéaire pour déterminer l’importance des variables ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# Importance = valeur absolue des coefficients
importance = np.abs(lr.coef_.ravel())

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/RL/diab_rl1.pdf")


# === Régression Linéaire avec toutes les variables ===
y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RL/diab_rl2.pdf")

# === Sélection des deux variables les plus importantes ===
top2_idx = np.argsort(importance)[-2:]
top2_features = X_train.columns[top2_idx]

# === Filtrage des données sur ces deux variables ===
X_train_top2 = X_train[top2_features]
X_test_top2 = X_test[top2_features]

# === Régression Linéaire ===
lr_top2 = LinearRegression()
lr_top2.fit(X_train_top2, y_train)

# === Prédiction ===
y_pred = lr_top2.predict(X_test_top2)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RL/diab_rl3.pdf")


# Cas mécanique Weight

entrees = list(range(9))
sorties = [0]
nombre = 1000

model = K_Mecanique(entrees, sorties, nombre)

data_full = model.donnees(nombre)
target_full = model.sortie(data_full)

# on prend les entrees et les sorties désirées de repectivement data et target en choississant les bonnes lignes
data = data_full[entrees]
target = target_full[sorties]
data = model.renommer(data,-1)
target = model.renommer(target,sorties[0])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# === Régression Linéaire pour déterminer l’importance des variables ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# Importance = valeur absolue des coefficients
importance = np.abs(lr.coef_.ravel())

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/RL/meca1_rl1.pdf")


# === Régression Linéaire avec toutes les variables ===
y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RL/meca1_rl2.pdf")

# === Sélection des deux variables les plus importantes ===
top2_idx = np.argsort(importance)[-2:]
top2_features = X_train.columns[top2_idx]

# === Filtrage des données sur ces deux variables ===
X_train_top2 = X_train[top2_features]
X_test_top2 = X_test[top2_features]

# === Régression Linéaire ===
lr_top2 = LinearRegression()
lr_top2.fit(X_train_top2, y_train)

# === Prédiction ===
y_pred = lr_top2.predict(X_test_top2)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RL/meca1_rl3.pdf")

# Cas mécanique VC_middle

entrees = list(range(9))
sorties = [6]
nombre = 1000

model = K_Mecanique(entrees, sorties, nombre)

data_full = model.donnees(nombre)
target_full = model.sortie(data_full)

# on prend les entrees et les sorties désirées de repectivement data et target en choississant les bonnes lignes
data = data_full[entrees]
target = target_full[sorties]
data = model.renommer(data,-1)
target = model.renommer(target,sorties[0])

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# === Régression Linéaire pour déterminer l’importance des variables ===
lr = LinearRegression()
lr.fit(X_train, y_train)

# Importance = valeur absolue des coefficients
importance = np.abs(lr.coef_.ravel())

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/RL/meca2_rl1.pdf")


# === Régression Linéaire avec toutes les variables ===
y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RL/meca2_rl2.pdf")

# === Sélection des deux variables les plus importantes ===
top2_idx = np.argsort(importance)[-2:]
top2_features = X_train.columns[top2_idx]

# === Filtrage des données sur ces deux variables ===
X_train_top2 = X_train[top2_features]
X_test_top2 = X_test[top2_features]

# === Régression Linéaire ===
lr_top2 = LinearRegression()
lr_top2.fit(X_train_top2, y_train)

# === Prédiction ===
y_pred = lr_top2.predict(X_test_top2)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RL/meca2_rl3.pdf")


# Random Forest
# Cas test diabétique

entrees = range(10)
model = K_Diabete(entrees)

data_full = model.donnees()
target_full = model.sortie()
data = data_full.copy()
target = target_full.copy()


# Split initial
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Modèle Random Forest initial
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.values.ravel())

# Importances des variables
importance = rf.feature_importances_
features = data.columns
sorted_idx = np.argsort(importance)[::-1]
top2_features = features[sorted_idx[:2]]

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/RF/diab_rf1.pdf")

# === Prédictions ===
y_pred = rf.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RF/diab_rf2.pdf")




print(f"Top 2 features sélectionnées : {top2_features.tolist()}")


# Choisir un arbre de la forêt (par exemple le premier)
estimator = rf.estimators_[0]

# Visualisation
plt.figure(figsize=(20, 10))
plot_tree(estimator,
          feature_names=data.columns,
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)  # max_depth limite la profondeur affichée
plt.savefig("../../Figures/Jeux de données simples/RF/diab_rf3.pdf", dpi=500)
plt.show()

# Nouveau dataset avec seulement les 2 features les plus importantes
data_reduced = data[top2_features]

# Nouveau split
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(data_reduced, target, test_size=0.2, random_state=42)

# Nouveau modèle
rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reduced.fit(X_train_red, y_train_red.values.ravel())

# Prédictions
y_pred = rf_reduced.predict(X_test_red)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RF/diab_rf3.pdf")


# Cas test mécanique Weight

entrees = list(range(9))
sorties = [0]
nombre = 1000

model = K_Mecanique(entrees, sorties, nombre)

data_full = model.donnees(nombre)
target_full = model.sortie(data_full)

# on prend les entrees et les sorties désirées de repectivement data et target en choississant les bonnes lignes
data = data_full[entrees]
target = target_full[sorties]
data = model.renommer(data,-1)
target = model.renommer(target,sorties[0])

# Split initial
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Modèle Random Forest initial
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.values.ravel())

# Importances des variables
importance = rf.feature_importances_
features = data.columns
sorted_idx = np.argsort(importance)[::-1]
top2_features = features[sorted_idx[:2]]

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/RF/meca1_rf1.pdf")

# === Prédictions ===
y_pred = rf.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RF/meca1_rf2.pdf")




print(f"Top 2 features sélectionnées : {top2_features.tolist()}")


# Choisir un arbre de la forêt (par exemple le premier)
estimator = rf.estimators_[0]

# Visualisation
plt.figure(figsize=(20, 10))
plot_tree(estimator,
          feature_names=data.columns,
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)  # max_depth limite la profondeur affichée
plt.savefig("../../Figures/Jeux de données simples/RF/meca1_rf3.pdf", dpi=500)
plt.show()

# Nouveau dataset avec seulement les 2 features les plus importantes
data_reduced = data[top2_features]

# Nouveau split
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(data_reduced, target, test_size=0.2, random_state=42)

# Nouveau modèle
rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reduced.fit(X_train_red, y_train_red.values.ravel())

# Prédictions
y_pred = rf_reduced.predict(X_test_red)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RF/meca1_rf3.pdf")

# Cas test mécanique VC_middle

entrees = list(range(9))
sorties = [6]
nombre = 1000

model = K_Mecanique(entrees, sorties, nombre)

data_full = model.donnees(nombre)
target_full = model.sortie(data_full)

# on prend les entrees et les sorties désirées de repectivement data et target en choississant les bonnes lignes
data = data_full[entrees]
target = target_full[sorties]
data = model.renommer(data,-1)
target = model.renommer(target,sorties[0])

# Split initial
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Modèle Random Forest initial
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.values.ravel())

# Importances des variables
importance = rf.feature_importances_
features = data.columns
sorted_idx = np.argsort(importance)[::-1]
top2_features = features[sorted_idx[:2]]

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/RF/meca2_rf1.pdf")

# === Prédictions ===
y_pred = rf.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RF/meca2_rf2.pdf")




print(f"Top 2 features sélectionnées : {top2_features.tolist()}")


# Choisir un arbre de la forêt (par exemple le premier)
estimator = rf.estimators_[0]

# Visualisation
plt.figure(figsize=(20, 10))
plot_tree(estimator,
          feature_names=data.columns,
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=3)  # max_depth limite la profondeur affichée
plt.savefig("../../Figures/Jeux de données simples/RF/meca2_rf3.pdf", dpi=500)
plt.show()

# Nouveau dataset avec seulement les 2 features les plus importantes
data_reduced = data[top2_features]

# Nouveau split
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(data_reduced, target, test_size=0.2, random_state=42)

# Nouveau modèle
rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reduced.fit(X_train_red, y_train_red.values.ravel())

# Prédictions
y_pred = rf_reduced.predict(X_test_red)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/RF/meca2_rf3.pdf")


# Information mutuelle

# Cas test diabétique

entrees = range(10)
model = K_Diabete(entrees)

data_full = model.donnees()
target_full = model.sortie()
data = data_full.copy()
target = target_full.copy()

# Split des données
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# === Information Mutuelle ===
importance = mutual_info_regression(X_train, y_train.values.ravel(), random_state=42)
mi_series = pd.Series(importance, index=data.columns)

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/IM/diab_im1.pdf")


# === Régression Linéaire  ===
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/IM/diab_im2.pdf")


# === Régression Linéaire avec deux features ===
X_train = X_train[mi_series.nlargest(2).index]
X_test = X_test[mi_series.nlargest(2).index]
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/IM/diab_im3.pdf")


# Cas test mécanique Weight

entrees = list(range(9))
sorties = [0]
nombre = 1000

model = K_Mecanique(entrees, sorties, nombre)

data_full = model.donnees(nombre)
target_full = model.sortie(data_full)

# on prend les entrees et les sorties désirées de repectivement data et target en choississant les bonnes lignes
data = data_full[entrees]
target = target_full[sorties]
data = model.renommer(data,-1)
target = model.renommer(target,sorties[0])

# Split des données
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# === Information Mutuelle ===
importance = mutual_info_regression(X_train, y_train.values.ravel(), random_state=42)
mi_series = pd.Series(importance, index=data.columns)

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/IM/meca1_im1.pdf")


# === Régression Linéaire  ===
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/IM/meca1_im2.pdf")


# === Régression Linéaire avec deux features ===
X_train = X_train[mi_series.nlargest(2).index]
X_test = X_test[mi_series.nlargest(2).index]
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/IM/meca1_im3.pdf")

# Cas test mécanique VC_middle

entrees = list(range(9))
sorties = [6]
nombre = 1000

model = K_Mecanique(entrees, sorties, nombre)

data_full = model.donnees(nombre)
target_full = model.sortie(data_full)

# on prend les entrees et les sorties désirées de repectivement data et target en choississant les bonnes lignes
data = data_full[entrees]
target = target_full[sorties]
data = model.renommer(data,-1)
target = model.renommer(target,sorties[0])

# Split des données
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# === Information Mutuelle ===
importance = mutual_info_regression(X_train, y_train.values.ravel(), random_state=42)
mi_series = pd.Series(importance, index=data.columns)

plot_importance(importance, X_train.columns, save_path="../../Figures/Jeux de données simples/IM/meca2_im1.pdf")


# === Régression Linéaire  ===
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/IM/meca2_im2.pdf")


# === Régression Linéaire avec deux features ===
X_train = X_train[mi_series.nlargest(2).index]
X_test = X_test[mi_series.nlargest(2).index]
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plot_prediction_scatter(y_test.values, y_pred, save_path="../../Figures/Jeux de données simples/IM/meca2_im3.pdf")