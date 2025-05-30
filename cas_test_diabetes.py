from sklearn.datasets import load_diabetes
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
# (exemple) from .cas_test_mécanique import K_Mécanique 

class K_Diabete:
    def __init__(self, entrees, random_state=42, test_size=0.2):
        self.entrees = entrees
        self.sorties = [0]
        self.random_state = random_state
        self.test_size = test_size
        self.type = 'Diabète'

    def donnees(self):
        # Charger les données
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)  # Colonnes spécifiées
        return X
    
    def sortie(self):
        data = load_diabetes()
        y = pd.DataFrame(data.target)
        return y
