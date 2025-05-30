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
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression

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
    importance_array = importance_array
    plt.figure(figsize=(8, 6))
    plt.bar(feature_names, importance_array, color='blue')
    plt.ylabel("Importance", fontweight='bold')
    plt.xlabel("Variables", fontweight='bold')
    plt.yticks(fontweight='bold')
    plt.xticks(fontweight='bold')
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



cas = "tuto"

entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)
 
data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in data_full.columns]]
target = target_full[[str(i) for i in target_full.columns]]

X_train, X_test, y_train, y_test = train_test_split(data, target["3"], test_size=0.2, random_state=42)
# Sortie est le maximum de la tension



# Info mutuelle

np.random.seed(0)
X = np.random.rand(1000, 3)
y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)

f_test, _ = f_regression(X, y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, y)
mi /= np.max(mi)

plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[:, i], y, edgecolor="black", s=20)
    plt.xlabel("$x_{}$".format(i + 1), fontsize=14,fontweight='bold')
    if i == 0:
        plt.ylabel("$y$", fontsize=14,fontweight='bold')
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16,fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
plt.savefig("../../Figures/Etat de l'art/Info_mutuelle1.pdf", dpi=500)
plt.show()



# Étape 1 : calcul de l'information mutuelle
feature_names = data.columns.tolist()
mi = mutual_info_regression(X_train, y_train.values.ravel(), random_state=42)

# Étape 2 : affichage des importances
plot_importance(mi, feature_names, save_path="../../Figures/Etat de l'art/importance_mutual_info.pdf")

# Étape 3 : sélection de la moitié des variables les plus importantes
n_features = len(mi)
n_top_features = n_features // 2
indices_top = np.argsort(mi)[-n_top_features:]  # indices des meilleures variables
selected_features = [feature_names[i] for i in indices_top]

print(f"\nVariables sélectionnées ({n_top_features}): {selected_features}")

# Étape 4 : régression linéaire sur les variables sélectionnées
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

reg = LinearRegression()
reg.fit(X_train_selected, y_train)
y_pred = reg.predict(X_test_selected)

# Étape 5 : affichage de la prédiction
plot_prediction_scatter(y_test, y_pred, save_path="../../Figures/Etat de l'art/prediction_mutual_info.pdf")



# Régression linéaire
# Étape 1 : entraînement d'une régression linéaire sur toutes les variables
feature_names = data.columns.tolist()

reg_full = LinearRegression()
reg_full.fit(X_train, y_train)

# Importance = valeur absolue des coefficients
coefs = np.abs(reg_full.coef_.ravel())

# Étape 2 : affichage des importances
plot_importance(coefs, feature_names, save_path="../../Figures/Etat de l'art/importance_regression_lineaire.pdf")

# Étape 3 : sélection de la moitié des variables les plus importantes
n_features = len(coefs)
n_top_features = n_features // 2
indices_top = np.argsort(coefs)[-n_top_features:]  # indices des meilleures variables
selected_features = [feature_names[i] for i in indices_top]

print(f"\nVariables sélectionnées ({n_top_features}) : {selected_features}")

# Étape 4 : régression linéaire sur les variables sélectionnées
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

reg = LinearRegression()
reg.fit(X_train_selected, y_train)
y_pred = reg.predict(X_test_selected)

# Étape 5 : affichage de la prédiction
plot_prediction_scatter(y_test, y_pred, save_path="../../Figures/Etat de l'art/prediction_regression_lineaire.pdf")



# Régression par forêt aléatoire
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Étape 1 : entraînement du modèle de forêt aléatoire
feature_names = data.columns.tolist()

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.values.ravel())

# Importance = feature_importances_ de la forêt
importances = rf.feature_importances_

# Étape 2 : affichage des importances
plot_importance(importances, feature_names, save_path="../../Figures/Etat de l'art/importance_random_forest.pdf")

# Étape 3 : sélection de la moitié des variables les plus importantes
n_features = len(importances)
n_top_features = n_features // 2
indices_top = np.argsort(importances)[-n_top_features:]  # indices des meilleures variables
selected_features = [feature_names[i] for i in indices_top]

print(f"\nVariables sélectionnées ({n_top_features}) : {selected_features}")

# Étape 4 : ré-entraîner une forêt aléatoire sur les variables sélectionnées
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

rf_selected = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train.values.ravel())
y_pred = rf_selected.predict(X_test_selected)

# Étape 5 : affichage de la prédiction
plot_prediction_scatter(y_test, y_pred, save_path="../../Figures/Etat de l'art/prediction_random_forest.pdf")


## ACP
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d  # noqa: F401

# Charger les données Iris
iris = load_iris(as_frame=True)
df = iris.frame.copy()

# Renommer les colonnes en français
df.columns = ["Longueur sépale", "Largeur sépale", "Longueur pétale", "Largeur pétale", "Cible"]
df["Cible"] = iris.target_names[iris.target]

# Créer le pairplot
g = sns.pairplot(df, hue="Cible")

# Mettre le texte en gras (axes, ticks, légende)
for ax in g.axes.flatten():
    if ax is not None:
        ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
        ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
        ax.tick_params(axis='both', labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

# Mettre en gras les titres de légende
if g._legend:
    for text in g._legend.texts:
        text.set_fontweight('bold')
    g._legend.set_title("Classes", prop={'weight': 'bold'})

    # Déplacer la légende à l'extérieur du graphique
    g._legend.set_bbox_to_anchor((1, 0.35))
    g._legend.set_loc("center right")

plt.tight_layout()
plt.savefig("../../Figures/Etat de l'art/paires_iris_français.pdf", dpi=500)
plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Charger les données
iris = load_iris(as_frame=True)

# PCA
pca = PCA(n_components=3)
X_reduit = pca.fit_transform(iris.data)
explained = pca.explained_variance_ratio_ * 100  # en pourcents

# Figure 3D
fig = plt.figure(1, figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

scatter = ax.scatter(
    X_reduit[:, 0],
    X_reduit[:, 1],
    X_reduit[:, 2],
    c=iris.target,
    s=40,
)

# Ajouter les pourcentages aux labels d'axes
ax.set_xlabel(f"1ère composante ({explained[0]:.1f}%)", fontweight='bold')
ax.set_ylabel(f"2ème composante ({explained[1]:.1f}%)", fontweight='bold')
ax.set_zlabel(f"3ème composante ({explained[2]:.1f}%)", fontweight='bold')

# Supprimer les ticks numériques
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Légende
legend = ax.legend(
    scatter.legend_elements()[0],
    iris.target_names.tolist(),
    loc="upper right",
    title="Classes"
)
ax.add_artist(legend)

# Noms des variables
feature_names = iris.feature_names

# Affichage des équations
for i, composante in enumerate(pca.components_):
    equation = " + ".join([f"{coef:.3f}·{name}" for coef, name in zip(composante, feature_names)])
    print(f"Composante {i+1} = {equation}")


# Sauvegarde
plt.tight_layout()
plt.savefig("../../Figures/Etat de l'art/importances_pca_3D.pdf", dpi=500)
plt.show()



from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np

# Étape 1 : entraînement PCA sur les données centrées et réduites
feature_names = data.columns.tolist()

pca = PCA()
X_pca = pca.fit_transform(X_train)

# Importance de chaque variable = combinaison pondérée des composantes
explained_var = pca.explained_variance_ratio_
importance = np.abs(pca.components_.T @ explained_var)

# Étape 2 : affichage des importances
plot_importance(importance, feature_names, save_path="../../Figures/Etat de l'art/importance_pca.pdf")

# Étape 3 : sélection de la moitié des variables les plus importantes
n_features = len(importance)
n_top_features = n_features // 2
indices_top = np.argsort(importance)[-n_top_features:]
selected_features = [feature_names[i] for i in indices_top]

print(f"\nVariables sélectionnées ({n_top_features}) : {selected_features}")

# Étape 4 : entraînement d'un modèle (ex. régression linéaire) sur les variables sélectionnées
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

reg = LinearRegression()
reg.fit(X_train_selected, y_train)
y_pred = reg.predict(X_test_selected)

# Étape 5 : affichage de la prédiction
plot_prediction_scatter(y_test, y_pred, save_path="../../Figures/Etat de l'art/prediction_pca.pdf")


