import pandas as pd
import numpy as np

class K_structuraux:
    entrees = None
    sorties = None

    def __init__(self, entrees, sorties, random_state=42, test_size=0.2):
        self.entrees = entrees
        self.sorties = sorties
        self.random_state = random_state
        self.test_size = test_size
        self.type = 'Structuraux'

    def donnees(self, cas):
        if cas == "tuto":
            file_in = "../../tuto_input_1k.csv"
        elif cas == "dome":
            file_in = "../../dome_input_30.csv"
        elif cas == "bldg":
            file_in = "../../bldg_input_30.csv"
        elif cas == "wing":
            file_in = "../../wing_input_30.csv"
        else:
            raise ValueError("Cas invalide")
        
        data = pd.read_csv(file_in)
        if cas == "tuto":
            data = data.head(300)
        else : 
            data = pd.read_csv(file_in)
        
        return data
    
    def sortie(self, cas):
        if cas == "tuto":
            file_out = "../../tuto_output_1k.csv"
        elif cas == "dome":
            file_out = "../../dome_output_30.csv"
        elif cas == "bldg":
            file_out = "../../bldg_output_30.csv"
        elif cas == "wing":
            file_out = "../../wing_output_30.csv"
        else:
            raise ValueError("Cas invalide")
        
        data = pd.read_csv(file_out)
        if cas == "tuto":
            data = data.head(300)
        else:
            data.columns = ['mass', 'max.stress', 'max.displacement', 'max.deflection']
        
        return data

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