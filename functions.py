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
from src.cas_test_diabetes import K_Diabete
from src.cas_test_mécanique import K_Mecanique
from sklearn.decomposition import PCA


def mutual_info(modele, data, target):
    """
    Calcule l'information mutuelle entre des caractéristiques continues et une cible continue.

    Args:
        modele: Model containing feature names in entrees attribute
        data (pd.DataFrame): DataFrame containing continuous features
        target (pd.Series): Series containing continuous target variable

    Returns:
        pd.DataFrame: DataFrame with feature names as index and mutual information values
    """
    # Calcul de l'information mutuelle 
    mi = np.zeros((data.shape[1], target.shape[1]))
    for i in range(target.shape[1]):
        mi[:, i] = mutual_info_regression(data, target.iloc[:, i], random_state=modele.random_state)    
    
    # Create DataFrame with feature names
    mi = pd.DataFrame(mi, index=modele.entrees, columns=target.columns)
    
    return mi


def top_variables_out(top, mi):
    """
    Retourne les n variables les plus importantes pour chaque cible.

    Args:
        top (int): Number of top variables to return
        mi (pd.DataFrame): DataFrame with feature names as index and mutual information values

    Returns:
        pd.DataFrame: DataFrame with feature names as index and mutual information values
    """
    
    # Retourne les n variables les plus importantes 
    mi = mi.apply(lambda x: x.nlargest(top).index)
    # change l'index pour aller de 0 à top
    return mi.reset_index(drop=True)

def top_variables_in(n, mi):
    """
    Pour chaque variable de sortie, renvoie les n variables d'entrée les plus importantes

    Args:
        n (int): Number of top variables to return
        mi (pd.DataFrame): DataFrame with feature names as index and mutual information values
    """
    if mi.shape[1] <= 1:
        raise ValueError("This function is only useful when there are multiple target variables")
    
    result = pd.DataFrame(index=range(n), columns=mi.index)
    
    for col in mi.index:
        # Get top N indices for this target
        top_indices = mi.loc[col].sort_values(ascending=False).index[:n]
        # Fill column with these indices
        result[col] = top_indices.values
        
    return result

def linear_feature_selection(model, data, target, treshold, top):
    """
    performs feature selection using linear regression and mutual information
    Args:
        model: Model containing feature names in entrees attribute
        data (pd.DataFrame): DataFrame containing continuous features
        target (pd.DataFrame): Series containing continuous target variable
        method (str): ???
        param (float): Parameter for percentage prediction method
        top (pd.DataFrame): DataFrame with feature names as index and mutual information values
    """
    # Division en __% entraînement et __% test
    X_train, X_test, y_train, y_test = train_test_split(data, pd.DataFrame(target), test_size=model.test_size, random_state=model.random_state)
    model_all = LinearRegression()
    model_all.fit(X_train, y_train)
    prediction = model_all.predict(X_test).reshape(-1, 1)  # Reshape to Nx1
    # On calcule le MSE entre prediction et y_test
    mse_all = np.mean((prediction - y_test)**2)
    # On fait la sélection de feature jusqu'à ce que le MSE soit à 90% du MSE avec toutes les features

    # Vérifier quelles variables atteignent __% de la performance
    selected_indices = []
    ok = False
    for i in top[0]:
        selected_indices.append(i)
        sub_model = LinearRegression()
        sub_model.fit(X_train[selected_indices], y_train)
        prediction = sub_model.predict(X_test[selected_indices]).reshape(-1, 1)  # Reshape to Nx1
        mse = np.mean((prediction - y_test)**2)
        r2 = r2_score(y_test, prediction)
        if mse < mse_all + (1-treshold) * mse_all:
            ok = True
            break
    return selected_indices, ok, mse, mse_all, r2
    
def RandomForest_feature_selection(model, data, target, treshold, top):
    """
    performs feature selection using Random Forest and mutual information
    Args:
        model: Model containing feature names in entrees attribute
        data (pd.DataFrame): DataFrame containing continuous features
        target (pd.DataFrame): Series containing continuous target variable
        method (str): ???
        param (float): Parameter for percentage prediction method
        top (pd.DataFrame): DataFrame with feature names as index and mutual information values
    """
    # Division en __% entraînement et __% test
    X_train, X_test, y_train, y_test = train_test_split(data, pd.DataFrame(target), test_size=model.test_size, random_state=model.random_state)
    model_all = RandomForestRegressor(n_estimators=100, random_state=model.random_state)
    model_all.fit(X_train, y_train.values.ravel())
    prediction = model_all.predict(X_test).reshape(-1, 1)  # Reshape to Nx1
    # On calcule le MSE entre prediction et y_test
    mse_all = np.mean((prediction - y_test)**2)
    # On fait la sélection de feature jusqu'à ce que le MSE soit à 90% du MSE avec toutes les features

    # Vérifier quelles variables atteignent __% de la performance
    selected_indices = []
    ok = False
    for i in top[0]:
        selected_indices.append(i)
        sub_model = RandomForestRegressor(random_state=model.random_state)
        sub_model.fit(X_train[selected_indices], y_train.values.ravel())
        prediction = sub_model.predict(X_test[selected_indices]).reshape(-1, 1)
        mse = np.mean((prediction - y_test)**2)
        r2 = r2_score(y_test, prediction)
        if mse < mse_all + (1-treshold) * mse_all:
            ok = True
            break
    return selected_indices, ok, mse, mse_all, r2

def top_variables_treshold(model, treshold, top, method, data, target):
    """
    Retourne les variables les plus importantes pour chaque cible en fonction d'un seuil atteint par le pouvoir prédictif.

    Args:
        treshold (float): Minimum R2 score ratio to achieve
        mi (pd.DataFrame): Mutual information matrix
        method (str): Selection method ("linear_regression")
        X (np.ndarray): Input data matrixa
        
    Returns:
        pd.DataFrame: DataFrame with feature names as index and mutual information values
    """
    if method == "linear_regression":
        if target.shape[1] == 1:
            selected_indices, ok, mse, mse_all, r2 = linear_feature_selection(model, data, target, treshold, top)
            print(f"target : mse : {np.sqrt(mse)}, r2 : {r2}")
            if ok == True:
                print(f"Le seuil de MSE_all : {mse_all:.3f} avec {treshold*100}% a été atteint avec MSE : {mse:.3f} et {len(selected_indices)} variables")
            return pd.DataFrame(selected_indices)
        else:
            #set the results to an array of dimension target.shape[1] x top at set it to -1
            results = np.full((top.shape[1] ,target.shape[1]), -1)
            j=0
            for i in target.columns.to_numpy():
                top_fct = pd.DataFrame(top[i].to_numpy())
                selected_indices, ok, mse, mse_all, r2 = linear_feature_selection(model, data, target[i], treshold, top_fct)
                print(f"{i} : mse : {np.sqrt(mse)}, r2 : {r2}")
                if ok == True:
                    print(f"Le seuil de MSE_all : {mse_all:.3f} avec {treshold*100}% a été atteint avec MSE : {mse:.3f} {len(selected_indices)} variables pour la sortie {i}")
                #on agrandit l'array selected indices pour qu'il soit de taille top.shape[1]
                selected_indices= np.pad(selected_indices, (0, top.shape[1]-len(selected_indices)), 'constant', constant_values=-1)
                # on met la variable selected_indices dans la première colonne de results
                results[:,j] = selected_indices
                j+=1
            return pd.DataFrame(results, columns=target.columns)
    if method == "RandomForest":
        if target.shape[1] == 1:
            selected_indices, ok, mse, mse_all, r2 = RandomForest_feature_selection(model, data, target, treshold, top)
            print(f"target : mse : {np.sqrt(mse)}, r2 : {r2}")
            if ok == True:
                print(f"Le seuil de MSE_all : {mse_all:.3f} avec {treshold*100}% a été atteint avec MSE : {mse:.3f} et {len(selected_indices)} variables")
            return pd.DataFrame(selected_indices)
        else:
            #set the results to an array of dimension target.shape[1] x top at set it to -1
            results = np.full((top.shape[1] ,target.shape[1]), -1)
            j=0
            for i in target.columns.to_numpy():
                top_fct = pd.DataFrame(top[i].to_numpy())
                selected_indices, ok, mse, mse_all, r2 = RandomForest_feature_selection(model, data, target[i], treshold, top_fct)
                print(f"{i} : mse : {np.sqrt(mse)}, r2 : {r2}")
                if ok == True:
                    print(f"Le seuil de MSE_all : {mse_all:.3f} avec {treshold*100}% a été atteint avec MSE : {mse:.3f} {len(selected_indices)} variables pour la sortie {i}")
                #on agrandit l'array selected indices pour qu'il soit de taille top.shape[1]
                selected_indices= np.pad(selected_indices, (0, top.shape[1]-len(selected_indices)), 'constant', constant_values=-1)
                # on met la variable selected_indices dans la première colonne de results
                results[:,j] = selected_indices
                j+=1
            return pd.DataFrame(results, columns=target.columns)
    else:
        raise ValueError("Unknown method")

    
def plot_efficacity(model, top, data, target, method):
    """
    Fais les graphes nécessaires

    Args: 
        model: Model containing feature names in entrees attribute
        data (pd.DataFrame): DataFrame containing continuous features
        target (pd.DataFrame): Series containing continuous target variable
        top (pd.DataFrame): DataFrame with top features. The index is the rank
    """
    # On prend les indices des features sélectionnées en enlevant les '-1' de la liste
    selected_indices = top[top != -1]
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=model.test_size, random_state=model.random_state)
    
    if method == "linear_regression" : 
        # Avec le top des features
        sub_model = LinearRegression()
        sub_model.fit(X_train[selected_indices], y_train)
        prediction = sub_model.predict(X_test[selected_indices])
        error = abs(prediction - y_test)
        # Avec toutes les features
        model_all = LinearRegression()
        model_all.fit(X_train, y_train)
        prediction_all = model_all.predict(X_test)
        error_all = abs(prediction_all - y_test)

    elif method =="RandomForest":
        # Avec le top des features
        sub_model = RandomForestRegressor(random_state=model.random_state)
        sub_model.fit(X_train[selected_indices], y_train.values.ravel())
        prediction = sub_model.predict(X_test[selected_indices])
        error = abs(prediction - y_test)
        # Avec toutes les features
        model_all = RandomForestRegressor(random_state=model.random_state)
        model_all.fit(X_train, y_train.values.ravel())
        prediction_all = model_all.predict(X_test)
        error_all = abs(prediction_all - y_test)


    # Plot the prediction and the actual values
    sns.scatterplot(x=y_test.index, y=y_test)
    sns.scatterplot(x=y_test.index, y=prediction, color='red')
    plt.title("RED : prediction, BLUE : actual values")
    plt.figure()
    sns.lineplot(x=error.index, y=error)
    plt.title(f"RMSE : {float(np.sqrt(np.mean(error**2))):.2f}, nombre de features : {len(selected_indices)}")

    # Plot to see the correlation between the prediction and the actual values
    plt.figure()
    sns.scatterplot(x=y_test, y=prediction)
    plt.title("Correlation between prediction and actual values")

    # Plot the prediction and the actual values
    plt.figure()
    sns.scatterplot(x=y_test.index, y=y_test)
    sns.scatterplot(x=y_test.index, y=prediction_all, color='red')
    plt.title("RED : prediction, BLUE : actual values")
    plt.figure()
    sns.lineplot(x=error_all.index, y=error_all)
    plt.title(f" RMSE : {float(np.sqrt(np.mean(error_all**2))):.2f}, nombre de features : {data.shape[1]}")

    # Plot to see the correlation between the prediction and the actual values
    plt.figure()
    sns.scatterplot(x=y_test, y=prediction_all)
    

def dynamic_mutual_info_out(model, data, target, indices, k_test):
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
    if k_test=="diabétique":
        data_dynamic = data.copy()
        for i in indices.dropna(): # On prend les indices qui ne sont pas des NaN
            data_dynamic[i] = 0
        data_dynamic = data_dynamic[model.entrees]
    else:
        mean = [1, 1, 1, 1, 1, 1, 1, 0.192, 0.192, 0, 0]
        data_dynamic = data.copy()
        for i in indices.dropna(): # On prend les indices qui ne sont pas des NaN
            data_dynamic[i] = mean[i] # On met la moyenne
        target = model.sortie(data_dynamic)
        data_dynamic = data_dynamic[model.entrees]
        target = target[model.sorties]    
    return mutual_info(model, data_dynamic, target)

def top_variables_out_dynamic(model, data_full, target_full, n):
    """
    Returns most influential variables by recalculating mutual information 
    after each variable selection.

    Args:
        model: Model containing feature names
        data_full: Input data
        target_full: Target data (single or multiple columns)
        n: Number of top variables to return

    Returns:
        pd.DataFrame: DataFrame with selected variables for each target
    """
    # Handle both single and multiple target cases
    targets = [0] if target_full.shape[1] == 1 else model.sorties
    
    result = pd.DataFrame(index=range(n), columns=targets)
    
    for col in targets:    
        for index in range(n):
            if target_full.shape[1] == 1:
                mi_dynamic = dynamic_mutual_info_out(model, data_full, target_full, result[col], "diabétique")
            else:
                mi_dynamic = dynamic_mutual_info_out(model, data_full, target_full, result[col], "mécanique")
            top_indice = mi_dynamic[col].idxmax()
            result.loc[index, col] = top_indice
            
    return result

def dynamic_mutual_info_in(model, data, target, selected_outputs, method):
    """
    Calculate mutual information with selected outputs set to zero
    Ici on change les sorties en les mettant à zéro une fois sélectionnées.

    """
    # Copy data to avoid modifying original
    target_dynamic = target.copy()
    
    # Set selected outputs to zero
    if not selected_outputs.empty:
        selected = selected_outputs.dropna()
        for idx in selected:
            target_dynamic.iloc[:, int(idx)] = 0
    data = data[model.entrees]
    target_dynamic = target_dynamic[model.sorties]
    return mutual_info(model, data, target_dynamic)

def top_variables_in_dynamic(model, data_full, target_full, n):
    """
    Returns most influenced outputs for each input variable by recalculating 
    mutual information after setting selected outputs to zero.

    Args:
        model: Model containing feature names
        data_full: Input data
        target_full: Target data
        n: Number of top outputs to return

    Returns:
        pd.DataFrame: DataFrame with selected outputs for each input
    """
    result = pd.DataFrame(index=range(n), columns=model.entrees)
    
    # Pour chaque entrée
    for col in model.entrees:    
        for index in range(n):
            # Calculate dynamic mutual information
            mi_dynamic = dynamic_mutual_info_in(model, data_full, target_full, result[col], "mécanique")
            # Get output with highest mutual information
            top_indice = mi_dynamic.loc[col].idxmax()
            # Store result
            result.loc[index, col] = top_indice
            
    return result


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def feature_selection_rf(data, target, entrees, test_size, random_state):

    # Initialiser et entraîner le modèle Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    rf.fit(data, target.values.ravel())
    # Récupérer l'importance des features
    feature_importances = rf.feature_importances_

    sorted_indices = np.argsort(feature_importances)[::-1]

    selected_features = [entrees[i] for i in sorted_indices]
    importance_feature = [feature_importances[i] for i in sorted_indices]
    result_df = pd.DataFrame({
        'feature': selected_features,
        'Gini importance': importance_feature
    })
    
    return result_df

def feature_selection_f_regression(data, target, entrees, test_size, random_state):
    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )
    
    # Calculer les scores F et les p-valeurs
    f_scores, p_values = f_regression(X_train, y_train.values.ravel())
    
    # Trier les features par score F décroissant
    sorted_indices = np.argsort(f_scores)[::-1]
    
    # Sélection des features triées
    selected_features = [entrees[i] for i in sorted_indices]
    importance_feature = [f_scores[i] for i in sorted_indices]
    result_df = pd.DataFrame({
        'feature': selected_features,
        'F-score': importance_feature
    })

    return result_df

def feature_selection_rfe(model, data, target, entrees, test_size, random_state, sub_model):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )
    
    if sub_model=="RandomForest":
        sub_model_new = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif sub_model=="LinearRegression":
        sub_model_new = LinearRegression()
    else:
        raise ValueError("Unknown model")
    
    selector = RFE(sub_model_new, n_features_to_select=5)
    selector.fit(X_train, y_train.values.ravel())
    selected_features = [entrees[i] for i in range(len(entrees)) if selector.support_[i]]
    

    rmse, r2 = model_fit(model, data, target, sub_model, selected_features)

    return pd.DataFrame(selected_features), rmse, r2

def feature_selection_selectfrommodel(model, data, target, entrees, test_size, random_state, sub_model):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )
    
    if sub_model=="RandomForest":
        sub_model_new = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif sub_model=="LinearRegression":
        sub_model_new = LinearRegression()
    else:
        raise ValueError("Unknown model")
    
    selector = SelectFromModel(sub_model_new, max_features=5, threshold="0.001*mean")
    selector.fit(X_train, y_train.values.ravel())
    selected_features = [entrees[i] for i in range(len(entrees)) if selector.get_support()[i]]

    rmse, r2 = model_fit(model, data, target, sub_model, selected_features)

    return pd.DataFrame(selected_features), rmse, r2

def feature_selection_sequential(model, data, target, entrees, test_size, random_state, direction, sub_model):
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )
    if sub_model=="RandomForest":
        sub_model_new = RandomForestRegressor(n_estimators=100, random_state=random_state)
    elif sub_model=="LinearRegression":
        sub_model_new = LinearRegression()
    else:
        raise ValueError("Unknown model")
    
    selector = SequentialFeatureSelector(sub_model_new, n_features_to_select=5, direction=direction)
    selector.fit(X_train, y_train)
    selected_features = [entrees[i] for i in range(len(entrees)) if selector.get_support()[i]]
    
    rmse, r2 = model_fit(model, data, target, sub_model, selected_features)

    return pd.DataFrame(selected_features), rmse, r2


def feature_selection_PCA(data, target,entrees, test_size, random_state, n_components):
    """
    Perform PCA on the data and return the top features
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_state
    )
    pca =  PCA(n_components=n_components)
    X_reduced =pca.fit_transform(X_train)

    return pd.DataFrame(pca.components_, columns=entrees)

def model_fit(model, data, target, method, features):
    """
    Fit a model to the data and target using the specified method.

    Args:
        model: Model containing feature names in entrees attribute
        data (pd.DataFrame): DataFrame containing continuous features
        target (pd.DataFrame): Series containing continuous target variable
        method (str): Method to use for fitting the model
    """
    X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=model.test_size, random_state=model.random_state
    )
    if method == "LinearRegression":
        sub_model = LinearRegression()
        sub_model.fit(X_train[features], y_train)
        if target.shape[1] == 1:
            prediction = sub_model.predict(X_test[features])
        else :     
            prediction = sub_model.predict(X_test[features]).reshape(-1, 1)
        rmse = np.sqrt(np.mean((prediction - y_test)**2))
        r2 = r2_score(y_test, prediction)
    elif method == "RandomForest":
        sub_model = RandomForestRegressor(n_estimators=100, random_state=model.random_state)
        sub_model.fit(X_train[features], y_train.values.ravel())
        prediction = sub_model.predict(X_test[features])
        rmse = np.sqrt(np.mean((prediction - y_test)**2))
        r2 = r2_score(y_test, prediction)
    else:
        raise ValueError(f"Unknown method : {method}")
    return rmse, r2




# def feature_selection(): 


# def subset_generation(search_strategy):
#     if search_strategy == "sequential":

#     else:
#         raise ValueError("Unknown search strategy")
    

# def evaluation_criterion(criterion):
