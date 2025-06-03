import numpy as np
import pandas as pd
from preprocess_cpv import extract_cpv_hierarchy_level, add_cpv_hierarchy_column


#retrait des marchés supérieurs à 50 millions et inférieur à 20 milles.
def drop_outliers(df):
    df.drop(df[df['montant'] > 50000000].index, inplace=True)
    df.drop(df[df['montant'] < 20000].index, inplace=True)
    return df


#ajout d'une colonne avec les 2 premiers chiffres du CPV, et les trois premiers si c'est 45 ou 71
def cpv_2et3 (df):

    df = add_cpv_hierarchy_column(df)
    df3 = add_cpv_hierarchy_column(df, level=3)
    df["codeCPV_3"] = df3["codeCPV_3"]
    if df["codeCPV_2"] == "45000000":
        df["codeCPV_2"] = df["codeCPV_3"]
    if df["codeCPV_2"] == "71000000":
        df["codeCPV_2"] = df["codeCPV_3"]
        
    return df
