from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, RFE, SelectFromModel, SequentialFeatureSelector
# from src.cas_test_diabetes import K_Diabete
# from src.cas_test_mécanique import K_Mecanique
from sklearn.decomposition import PCA
import subprocess


    

def error(prediction, y_test, measure):
    """
    Fonction qui calcule l'erreur entre les prédictions et les vraies valeurs
        prediction : les valeurs prédites
        y_test : les vraies valeurs
        measure : la métrique utilisée pour calculer l'erreur

        return : la valeur de l'erreur
    """
    
    if measure == 'R2':
        return r2_score(y_test, prediction)
    elif measure == 'MSE':
        return np.mean((y_test - prediction)**2)
    else:
        raise ValueError('Métrique non reconnue')

def model_fit(model, estimator, X_train, y_train, X_test, y_test, method):
    """
    Fonction qui entraîne un modèle sur un jeu de données
        model : le cas test considéré
        X_train, y_train : les données d'entraînement avec seulement les entrées et les sorties qui nous intéressent
        X_test, y_test : les données de test avec seulement les entrées et les sorties qui nous intéressent
        method : la méthode utilisée pour l'entraînement [(All, Specific, Mean),([sortie à prédire]),(métrique)]

        return : un DataFrame avec le nom des sorties en index et la métrique en colonne
    """
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=model.random_state)
    measure = []
    if method[0] == 'All':
        for sorties in y_train.columns:
            estimator.fit(X_train, y_train[sorties])
            y_pred = estimator.predict(X_test)
            measure.append(error(y_pred, y_test[sorties], method[-1]))
        return pd.DataFrame(measure, columns=[method[-1]], index=y_train.columns)
    
    elif method[0] == 'Specific':
        for sorties in method[1]:
            estimator.fit(X_train, y_train[sorties])
            y_pred = estimator.predict(X_test)
            measure.append(error(y_pred, y_test[sorties], method[-1]))
        return pd.DataFrame(measure, columns=[method[-1]], index=method[1])
    
    elif method[0]== 'Mean':
        method[0] = 'All'
        df = model_fit(model, estimator, X_train, y_train, X_test, y_test, method)
        measure.append(df.mean())
        return pd.DataFrame(measure, columns=[method[-1]], index=['Mean'])
    else:
        raise ValueError('Méthode non reconnue')


def stopping_criterion(model, X_train, y_train, X_test, y_test, stopping_criterion, estimator, subset, iteration, MI_arg):
    """
    Vérifie si les variables sélectionnées prédisent aussi bien que toutes les variables
        model : le cas test considéré
        X_train, y_train : les données d'entraînement avec seulement les entrées et les sorties qui nous intéressent
        X_test, y_test : les données de test avec seulement les entrées et les sorties qui nous intéressent
        stopping_criterion : le critère d'arrêt [(subset_max, iter_max, R2, MSE), (5, 100, 0.9, 0.1), (any, all), (All, Specific, Mean), ([1,3])]
        estimator : l'estimateur utilisé pour l'entraînement
        subset : les variables sélectionnées par l'algorithme jusqu'à présent

        return : True si le critère d'arrêt est atteint, False sinon
    """
    
    # Estimateur
    if estimator =='RandomForest':
        estimator = RandomForestRegressor(n_estimators=100, random_state=model.random_state, bootstrap=True)
    elif estimator == 'LinearRegression':
        estimator = LinearRegression()
    elif MI_arg == 0:
        estimator = LinearRegression()
    elif MI_arg == 1:
        estimator = RandomForestRegressor(n_estimators=100, random_state=model.random_state, bootstrap=True)
    else :
        raise ValueError("Estimateur non reconnu")
    
    # Critères d'arrêt
    if stopping_criterion[0] == 'subset_max':
        if len(subset) == stopping_criterion[1]:
            return True
        else:
            return False
        
    elif stopping_criterion[0] == 'iter_max':
        if iteration == stopping_criterion[1]:
            return True
        else:
            return False
        
    elif stopping_criterion[0] == 'R2':
        if stopping_criterion[2] == 'any':
            if stopping_criterion[3] == 'All':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['All', 'R2'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['All', 'R2'])

            elif stopping_criterion[3] == 'Specific':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['Specific', stopping_criterion[4], 'R2'])
                # print("error_model_all : \n", error_model_all)
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['Specific', stopping_criterion[4], 'R2'])
                # print("error_model_subset: \n", error_model_subset)

            elif stopping_criterion[3] == 'Mean':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['Mean', 'R2'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['Mean', 'R2'])
            else:
                raise ValueError('Missing argument in stopping_criterion')
            
            if (error_model_subset.values >= stopping_criterion[1]*error_model_all.values).any():
                return True
            else:
                return False

        elif stopping_criterion[2] == 'all':
            if stopping_criterion[3] == 'All':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['All', 'R2'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['All', 'R2'])

            elif stopping_criterion[3] == 'Specific':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['Specific', stopping_criterion[4], 'R2'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['Specific', stopping_criterion[4], 'R2'])
            
            elif stopping_criterion[3] == 'Mean':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['Mean', 'R2'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['Mean', 'R2'])
            else:
                raise ValueError('Missing argument in stopping_criterion')

            if (error_model_subset.values >= stopping_criterion[1]*error_model_all.values).all():
                return True
            else:
                return False
        else:
            raise ValueError('Missing argument in stopping_criterion')

    elif stopping_criterion[0] == 'MSE':
        if stopping_criterion[2] == 'any':
            if stopping_criterion[3] == 'All':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['All', 'MSE'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['All', 'MSE'])

            elif stopping_criterion[3] == 'Specific':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['Specific', stopping_criterion[4], 'MSE'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['Specific', stopping_criterion[4], 'MSE'])
            
            elif stopping_criterion[3] == 'Mean':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['Mean', 'MSE'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['Mean', 'MSE'])
            else:
                raise ValueError('Missing argument in stopping_criterion')
            
            if (error_model_subset.values <= (1+stopping_criterion[1])*error_model_all.values).any():
                return True
            else:
                return False

        elif stopping_criterion[2] == 'all':
            if stopping_criterion[3] == 'All':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['All', 'MSE'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['All', 'MSE'])

            elif stopping_criterion[3] == 'Specific':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['Specific', stopping_criterion[4], 'MSE'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['Specific', stopping_criterion[4], 'MSE'])
            
            elif stopping_criterion[3] == 'Mean':
                error_model_all = model_fit(model, estimator, X_train, y_train, X_test, y_test, ['Mean', 'MSE'])
                error_model_subset = model_fit(model, estimator, X_train[subset], y_train, X_test[subset], y_test, ['Mean', 'MSE'])
            else:
                raise ValueError('Missing argument in stopping_criterion')

            if (error_model_subset.values <= (1+stopping_criterion[1])*error_model_all.values).all():
                return True
            else:
                return False
        else:
            raise ValueError('Missing argument in stopping_criterion')
    else:
       raise ValueError('Critère d\'arrêt non reconnu')
    
def mutual_info(model, X_train, y_train, mode):
    """
    Calcule l'information mutuelle entre des caractéristiques continues et une cible continue.

        modele: Model containing feature names in entrees attribute
        data (pd.DataFrame): DataFrame containing continuous features
        target (pd.Series): Series containing continuous target variable

    return:  information mutuelle entre les caractéristiques et la cible
        
    """
    if mode ==0:
        mi = mutual_info_regression(X_train, y_train, random_state=model.random_state, n_neighbors=100)
    elif mode == 1:
        # Calcul de l'information mutuelle 
        mi = np.zeros((X_train.shape[1], y_train.shape[1]))
        for i in range(y_train.shape[1]):
            mi[:, i] = mutual_info_regression(X_train, y_train.iloc[:, i].values.ravel(), random_state=model.random_state, n_neighbors=100)    
    else:
        raise ValueError('Mode non reconnu')
    return mi


def dynamic_mutual_info_out(model, X_train, y_train, sortie, done, data_full, target_full):
    """
    Calcule l'information mutuelle entre des caractéristiques continues et une cible continue.
    Ici on change les données en mettant la moyenne des valeurs des features sélectionnées.
    Args:
        modele: Model containing feature names in entrees attribute
        data (pd.DataFrame): DataFrame containing continuous features
        target (pd.Series): Series containing continuous target variable
        indices (pd.DataFrame): DataFrame with indices of the features to be averaged

    Returns:
        pd.DataFrame: DataFrame with feature names as index and mutual information values
    """
    if model.type=="Diabète":
        data_dynamic = X_train.copy()
        for i in done: # On prend les indices qui ne sont pas des NaN
            data_dynamic[i] = 0
        data_dynamic = data_dynamic[model.entrees]
        mi = pd.DataFrame(mutual_info(model, data_dynamic, y_train, 0), index=model.entrees, columns=[sortie])
        return mi.idxmax()[sortie]
    
    elif model.type=="Mécanique":
        mean = [1, 1, 1, 1, 1, 1, 1, 0.192, 0.192, 0, 0]
        data_dynamic = data_full.copy()
        for i in done: # On prend les indices qui ne sont pas des NaN
            data_dynamic[i] = mean[i] # On met la moyenne
        target_dynamic = model.sortie(data_dynamic)[sortie]
        data_dynamic = data_dynamic[model.entrees]

        mi = pd.DataFrame(mutual_info(model, data_dynamic, target_dynamic, 0), index=model.entrees, columns=[sortie])

        return mi.idxmax()[sortie]
    else:
        raise ValueError('Type de modèle non reconnu')

def estimator_fit_importance(model, X_train, y_train, estimator, sortie, data_full, target_full):
    """
    Fonction qui entraîne un estimateur sur un jeu de données et donne l'importance des variables
        model : le cas test considéré
        X_train, y_train : les données d'entraînement avec seulement les entrées et LA sortie qui nous intéressent
        estimator : l'estimateur utilisé pour l'entraînement
    
        return : un vecteur avec le nom des sorties en index et la métrique en colonne
    """

    if estimator == 'RandomForest':
        estimator = RandomForestRegressor(n_estimators=100, random_state=model.random_state, bootstrap=True)
        estimator.fit(X_train, y_train)
        importance = estimator.feature_importances_

    elif estimator == 'LinearRegression':
        estimator = LinearRegression()
        estimator.fit(X_train, y_train)
        importance = np.abs(estimator.coef_)

    elif estimator == 'PCA':
        estimator = PCA()
        estimator.fit(X_train)
        
        # Importance des features basée sur la somme des valeurs absolues des loadings pondérées par la variance expliquée
        importance = np.abs(estimator.components_.T @ estimator.explained_variance_ratio_)

    elif estimator == 'MI':
        importance = mutual_info(model, X_train, y_train, 0)

    elif estimator == 'DynMI':
        done = []
        importance = np.full(X_train.shape[1], -1)
        position = 0
        for indices in X_train.columns:
            feature = dynamic_mutual_info_out(model, X_train, y_train, sortie, done, data_full, target_full)
            if feature not in done : 
                importance[X_train.columns.get_loc(feature)] = position
            else : 
                for i in range(len(importance)) : 
                    if importance[i] == -1 : importance[i] = position
                break
            done.append(feature)
            position+=1
        
    else:
        raise ValueError('Estimateur non reconnu')
    return importance


def evalutation_criterion(model, X_train, y_train, evaluation_criterion, data_full, target_full):
    """
    Fonction qui trie l'importance des variables selon un estimateur

        model : le cas test considéré
        estimator : l'estimateur utilisé pour l'évaluation
        evaluation_criterion = les critères d'évaluation [(RandomForest, LinearRegression, PCA, MI, DynMI), (All, Specific, Mean), ([1,3])]

        return : un DataFrame avec les variables en colonne et leur importance en index

    """

    if evaluation_criterion[1] == 'All':
        for sorties in y_train.columns:
            importance = estimator_fit_importance(model, X_train, y_train[sorties], evaluation_criterion[0], sorties, data_full, target_full)
            importance = pd.DataFrame(importance, columns=[sorties], index=X_train.columns)

            if sorties == y_train.columns[0]:
                importance_final = importance
            else:
                importance_final = pd.concat([importance_final, importance], axis=1)

    elif evaluation_criterion[1] == 'Specific':
        for sorties in evaluation_criterion[2]:
            importance = estimator_fit_importance(model, X_train, y_train[sorties], evaluation_criterion[0], sorties, data_full, target_full)
            importance = pd.DataFrame(importance, columns=[sorties], index=X_train.columns)

            if sorties == evaluation_criterion[2][0]:
                importance_final = importance
            else:
                importance_final = pd.concat([importance_final, importance], axis=1)
    
    elif evaluation_criterion[1] == 'Mean':
        evaluation_criterion[1] = 'All'
        importance_final = evalutation_criterion(model, X_train, y_train, evaluation_criterion, data_full, target_full)
        importance_final = pd.DataFrame(importance_final.mean(axis=1), columns=['Mean'], index=X_train.columns)
    
    return importance_final

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
    


def subset_generation(model, evaluation_params, stopping_params, data, target, data_full, target_full, MI_arg):
    """
    Génère un sous ensemble de variables selon un critère d'évaluation, un critère d'arrêt et un ensemble de variables d'entrées et de sorties
    Args:
        model: le cas test considéré
        evaluation_params: les critères d'évaluation [(RandomForest, LinearRegression, PCA, MI, DynMI), (All, Specific, Mean), ([1,3])]
        stopping_params: le critère d'arrêt [(subset_max, iter_max, R2, MSE), (5, 100, 0.9, 0.1), (any, all), (All, Specific, Mean), ([1,3])]
        data_full: les données d'entrée complètes
        target_full: les données de sortie complètes

    Returns:
        pd.DataFrame: DataFrame with columns for each target and rows containing selected features
    """
    # Initialize dictionary to store subsets for each column
    subsets = {col: [] for col in target.columns}
    # Ajout à subset d'une colonne 'Mean' si on choisit la moyenne
    if evaluation_params[1] == 'Mean':
        subsets['Mean'] = []
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=model.random_state)
    
    feature_importance = evalutation_criterion(model, X_train, y_train, evaluation_params, data_full, target_full)
    
    if MI_arg == 'LinearRegression':
        MI_arg = 0
    elif MI_arg =='RandomForest':
        MI_arg = 1

    
    boolean_ascending = False
    if evaluation_params[0] == 'DynMI':
        boolean_ascending = True

    # Process each target column separately
    for column in feature_importance.columns:
        # Sort features by importance for current column
        sorted_features = feature_importance[column].sort_values(ascending=boolean_ascending).index
        iteration = 0
        
        # Build subset for current column
        for feature in sorted_features:
            # Check if we should add this feature based on stopping criterion
            iteration += 1
            subsets[column].append(feature)
            if stopping_criterion(model, X_train, y_train, X_test, y_test, stopping_params, evaluation_params[0], subsets[column], iteration, MI_arg):
                break    
    
    # Convert to DataFrame with consistent size by padding with None
    max_len = max(len(subset) for subset in subsets.values())
    padded_subsets = {col: subset + [None] * (max_len - len(subset)) for col, subset in subsets.items()}
    
    return pd.DataFrame(padded_subsets)


def validate_results(model, estimator, data, target, data_full, target_full, MI_arg, subset, voir):
    """
    Fonction qui calcule le R2 et le MSE pour le modèle considéré avec le sous-ensemble de variables contenu dans subset.
    Renvoie un DataFrame avec les scores R2 et MSE en colonnes et les sorties en index.
    """
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=model.random_state)
    
    # Initialisation de l'estimateur
    if estimator == 'RandomForest':
        estimator = RandomForestRegressor(n_estimators=100, random_state=model.random_state, bootstrap=True)
    elif estimator == 'LinearRegression':
        estimator = LinearRegression()
    else:
        raise ValueError("Estimateur non reconnu")
    
    measure = pd.DataFrame(index=y_test.columns, columns=['R2', 'MSE'])
    
    for sortie in y_test.columns:
        estimator.fit(X_train[subset], y_train[sortie])  
        y_pred = estimator.predict(X_test[subset])  # Ajout de [subset] pour assurer la cohérence
        # Stocke les résultats dans le DataFrame
        measure.loc[sortie, 'R2'] = error(y_pred, y_test[sortie], 'R2')
        print(error(y_pred, y_test[sortie], 'R2'))
        measure.loc[sortie, 'MSE'] = error(y_pred, y_test[sortie], 'MSE')
    
    if voir !=-1 :
        estimator.fit(X_train[subset], y_train[voir])
        y_pred = estimator.predict(X_test[subset])
        print("Plot pour la sortie", voir , "\n")
        plot_predictions_vs_actuals(y_test[voir], y_pred)
    return measure



def plot_predictions_vs_actuals(y_test, y_pred):
    """
    Fonction qui trace le graphe entre les prédictions et les véritables valeurs pour une sortie donnée.

    Paramètres :
    - y_test : valeurs réelles
    - y_pred : valeurs prédites
    - sortie : nom de la sortie affichée sur le graphique

    Affiche un scatter plot avec une droite y = x pour comparer les valeurs réelles et prédites.
    """
    
    # Tracé du graphique
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, label="Prédictions")
    
    # Ajout de la droite y = x pour voir l'alignement parfait
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Idéal : y = x")
    
    # Personnalisation du graphique
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title("Comparaison Prédictions vs Réel")
    plt.legend()
    plt.grid(True)
    plt.show()



