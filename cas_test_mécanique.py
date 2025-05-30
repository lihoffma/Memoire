import numpy as np
from scipy.optimize import minimize
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

class K_Mecanique:
    entrees = None
    sorties = None

    def __init__(self, entrees, sorties, nombre, random_state=42, test_size=0.2):
        self.entrees = entrees
        self.sorties = sorties
        self.random_state = random_state  # Stocker le random_state
        self.test_size = test_size
        self.type = 'Mécanique'
        self.nombre = nombre

    def donnees(self, nombre):
        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        # Parameters for x_1 to x_7
        means = np.array([1.0] * 7) 
        std_devs = np.array([0.03] * 7)
        lower_bounds = np.array([0.5] * 7)
        upper_bounds = np.array([1.5] * 7)

        # Generate x_1 to x_7 
        x_i = np.random.normal(means, std_devs, (nombre, 7))
        x_i = np.clip(x_i, lower_bounds, upper_bounds)

        # Generate other columns
        x_8 = np.random.choice([0.192, 0.345], nombre)
        x_9 = np.random.choice([0.192, 0.345], nombre)
        x_10 = np.random.uniform(-30, 30, nombre)
        x_11 = np.random.uniform(-30, 30, nombre)

        # Stack all columns together
        data = np.column_stack((x_i, x_8, x_9, x_10, x_11))
        
        # Create DataFrame with columns names
        df = pd.DataFrame(data, columns=['x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11'])
        
        return df

    def Weight(self, x):
        return (
            1.98
            + 4.90 * x[0]
            + 6.67 * x[1]
            + 6.98 * x[2]
            + 4.01 * x[3]
            + 1.78 * x[4]
            + 2.73 * x[6]
        )

    def F_Abdom(self, x):
        return (
            1.16
            - 0.3717 * x[1] * x[3]
            - 0.00931 * x[1] * x[9]
            - 0.484 * x[2] * x[8]
            + 0.01343 * x[5] * x[9]
        )

    def Def_rib_l(self, x):
        return 46.36 - 9.9 * x[1] - 12.9 * x[0] * x[7] + 0.1107 * x[2] * x[9]

    def Def_rib_m(self, x):
        return (
            33.86
            + 2.95 * x[2]
            + 0.1792 * x[9]
            - 5.057 * x[0] * x[1]
            - 11.0 * x[1] * x[7]
            - 0.0215 * x[4] * x[9]
            - 9.98 * x[6] * x[7]
            + 22.0 * x[7] * x[8]
        )

    def Def_rib_u(self, x):
        return (
            28.98
            + 3.818 * x[2]
            - 4.2 * x[0] * x[1]
            + 0.0207 * x[4] * x[9]
            + 6.63 * x[5] * x[8]
            - 7.7 * x[6] * x[7]
            + 0.32 * x[8] * x[9]
        )

    def VC_upper(self, x):
        return (
            0.261
            - 0.0159 * x[0] * x[1]
            - 0.188 * x[0] * x[7]
            - 0.019 * x[1] * x[6]
            + 0.0144 * x[2] * x[4]
            + 0.0008757 * x[4] * x[9]
            + 0.08045 * x[5] * x[8]
            + 0.00139 * x[7] * x[10]
            + 0.00001575 * x[9] * x[10]
        )

    def VC_middle(self, x):
        return (
            0.214
            + 0.00817 * x[4]
            - 0.131 * x[0] * x[7]
            - 0.0704 * x[0] * x[8]
            + 0.03099 * x[1] * x[5]
            - 0.018 * x[1] * x[6]
            + 0.0208 * x[2] * x[7]
            + 0.121 * x[2] * x[8]
            - 0.00364 * x[4] * x[5]
            + 0.0007715 * x[4] * x[9]
            - 0.0005354 * x[5] * x[9]
            + 0.00121 * x[7] * x[10]
            + 0.00184 * x[8] * x[9]
            - 0.02 * x[1] * x[1]
        )

    def VC_lower(self, x):
        return (
            0.74
            - 0.61 * x[1]
            - 0.163 * x[2] * x[7]
            + 0.001232 * x[2] * x[9]
            - 0.166 * x[6] * x[8]
            + 0.227 * x[1] * x[1]
        )

    def Force_public(self, x):
        return (
            4.72
            - 0.5 * x[3]
            - 0.19 * x[1] * x[2]
            - 0.0122 * x[3] * x[9]
            + 0.009325 * x[5] * x[9]
            + 0.000191 * x[10] * x[10]
        )

    def Vel_B_pillar(self, x):
        return (
            10.58
            - 0.674 * x[0] * x[1]
            - 1.95 * x[1] * x[7]
            + 0.02054 * x[2] * x[9]
            - 0.0198 * x[3] * x[9]
            + 0.028 * x[5] * x[9]
        )

    def Vel_door(self, x):
        return (
            16.45
            - 0.489 * x[2] * x[6]
            - 0.843 * x[4] * x[5]
            + 0.0432 * x[8] * x[9]
            - 0.0556 * x[8] * x[10]
            - 0.000786 * x[10] * x[10]
        )
    # Pour chaque colonne du dataframe x, on calcule la sortie correspondante et on renvoie le tout dans un dataframe
    def sortie(self, x):
        output_functions = [
            self.Weight,
            self.F_Abdom, 
            self.Def_rib_l,
            self.Def_rib_m,
            self.Def_rib_u,
            self.VC_upper,
            self.VC_middle,
            self.VC_lower,
            self.Force_public,
            self.Vel_B_pillar,
            self.Vel_door
        ]
        x.columns = range(x.shape[1])
        results = np.zeros((len(x), len(output_functions)))
        for j, func in enumerate(output_functions):
            results[:, j] = func(x)
            
        return pd.DataFrame(results, columns=list(range(len(output_functions))))

        
        

    def fonction_objectif(self, x):  # Definition de la fonction objectif
        return self.Weight(x)  # La fonction Weight est la fonction objectif

    def contraintes(self, x):  # Definition des contraintes
        return {
            "contrainte1": 1.0 - self.F_Abdom(x),
            "contrainte2_l": 32.0 - self.Def_rib_l(x),
            "contrainte2_m": 32.0 - self.Def_rib_m(x),
            "contrainte2_u": 32.0 - self.Def_rib_u(x),
            "contrainte3_u": 0.32 - self.VC_upper(x),
            "contrainte3_m": 0.32 - self.VC_middle(x),
            "contrainte3_l": 0.32 - self.VC_lower(x),
            "contrainte4_public": 4.0 - self.Force_public(x),
            "contrainte4_Abdom": 4.0 - self.F_Abdom(x),
            "contrainte5": 9.9 - self.Vel_B_pillar(x),
            "contrainte6": 15.7 - self.Vel_door(x),
        }

    def optimiser(self, x, methode):  # Fonction d'optimisation
        # Ensemble des contraintes
        constraints = [
            {"type": "ineq", "fun": lambda x: 1.0 - self.F_Abdom(x)},
            {"type": "ineq", "fun": lambda x: 32.0 - self.Def_rib_l(x)},
            {"type": "ineq", "fun": lambda x: 32.0 - self.Def_rib_m(x)},
            {"type": "ineq", "fun": lambda x: 32.0 - self.Def_rib_u(x)},
            {"type": "ineq", "fun": lambda x: 0.32 - self.VC_upper(x)},
            {"type": "ineq", "fun": lambda x: 0.32 - self.VC_middle(x)},
            {"type": "ineq", "fun": lambda x: 0.32 - self.VC_lower(x)},
            {"type": "ineq", "fun": lambda x: 4.0 - self.Force_public(x)},
            {"type": "ineq", "fun": lambda x: 4.0 - self.F_Abdom(x)},
            {"type": "ineq", "fun": lambda x: 9.9 - self.Vel_B_pillar(x)},
            {"type": "ineq", "fun": lambda x: 15.7 - self.Vel_door(x)},
        ]

        # Borne des variables
        bounds = [
            (0.5, 1.5),  # x1
            (0.5, 1.5),  # x2
            (0.5, 1.5),  # x3
            (0.5, 1.5),  # x4
            (0.5, 1.5),  # x5
            (0.5, 1.5),  # x6
            (0.5, 1.5),  # x7
            (0.192, 0.345),  # x8
            (0.192, 0.345),  # x9
            (-30, 30),  # x10
            (-30, 30),  # x11
        ]
        x0 = x

        # Ajuster x8 et x9 à des valeurs discrètes si nécessaire
        # Sinon, laissez-les comme variables continues dans l'intervalle [0.192, 0.345] 
        solution = minimize(
            self.fonction_objectif,
            x0,
            method=methode,
            bounds=bounds,
            constraints=constraints,
        )
        if solution.success:
            print("Optimisation réussie :", solution)
            return solution
        else:
            print("Échec de l'optimisation :", solution.message)

    def renommer(self, df, a):
        # Renommer les colonnes du DataFrame
        if a == -1 :
            df.columns = [
                "x_1",
                "x_2",
                "x_3",
                "x_4",
                "x_5",
                "x_6",
                "x_7",
                "x_8",
                "x_9"
            ]
        else : 
            col = [
                "Weight",
                "F_Abdom",
                "Def_rib_l",
                "Def_rib_m",
                "Def_rib_u",
                "VC_upper",
                "VC_middle",
                "VC_lower",
                "Force_public",
                "Vel_B_pillar",
                "Vel_door",
            ]
            df.columns = [col[a]]
        return df
    
def choix(cas):
    # TUTO
    """
    10 input entre 0 et 1
    12 output 
    """
    # DOME
    """
    696 input entre 0 et 1
    4 output 
    """

    #BLDG
    """
    942 input entre 0 et 1
    4 output 
    """

    #WING
    """
    162 input entre 0 et 1
    4 output 
    """

    if cas =="tuto":
        return [range(10), list(filter(lambda x: x not in [5, 6, 9], range(12)))]
    elif cas == "dome":
        return [range(696), range(4)]
    elif cas == "bldg":
        return [range(942), range(4)]
    elif cas == "wing":
        return [range(162), range(4)]
    else:
        raise ValueError("Cas invalide")
    
