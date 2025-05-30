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

winsound.Beep(1000, 2000)  

# Groupement de variables via l'information mutuelle


cas = "wing"

entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in data_full.columns]]
target = target_full[[str(i) for i in target_full.columns]]


# Optimisation par cluster d'information mutuelle
sort = ['mass', 'max. stress', 'max. deflection']
# === Données ===
cas = "bldg"

entrees = choix(cas)[0]     # Pour prendre toutes les entrées et toutes les sorties
sorties = choix(cas)[1]

model = K_structuraux(entrees, sorties)

data_full = model.donnees(cas)
target_full = model.sortie(cas)

data = data_full[[str(i) for i in data_full.columns]]
target = target_full[[str(i) for i in target_full.columns]]

X_train, X_test, y_train, y_test = train_test_split(data, target[sort], test_size=0.2, random_state=42)

# Calcul de l'information mutuelle avec n_neighbors=169 pour 0, 1 et 3
mi_df0 = mutual_info_regression(X_train, y_train["mass"], n_neighbors=math.ceil(0.4*X_train.shape[1]), random_state=model.random_state)
mi_df1 = mutual_info_regression(X_train, y_train["max. stress"], n_neighbors=math.ceil(0.4*X_train.shape[1]), random_state=model.random_state)
mi_df3 = mutual_info_regression(X_train, y_train["max. deflection"], n_neighbors=math.ceil(0.4*X_train.shape[1]), random_state=model.random_state)

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
    if col not in ['mass', 'max. stress', 'max. deflection']:
        continue
    reg = LinearRegression()
    reg.fit(X_train_grouped, y_train[col])
    y_pred = reg.predict(X_test_grouped)
    r2 = r2_score(y_test[col], y_pred)
    r2_scores[col] = r2
    print(f"✅ R² pour la sortie {col} : {r2:.4f}")
