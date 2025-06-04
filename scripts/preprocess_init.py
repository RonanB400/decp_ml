import numpy as np
import pandas as pd
from preprocess_cpv import extract_cpv_hierarchy_level, add_cpv_hierarchy_column


#selection des colonnes
def columns_selection(df, cat):
    if cat == 'pred_montant':
        df = df[['procedure', 'dureeMois','nature', 'formePrix', 'offresRecues', 'ccag',
            'sousTraitanceDeclaree', 'typeGroupementOperateurs', 'tauxAvance',
            'origineFrance', 'idAccordCadre', 'dateNotification', 'marcheInnovant', 'codeCPV']]
        return df
    elif cat == 'marche_sim':
        df = df[['procedure', 'dureeMois','nature', 'formePrix', 'offresRecues', 'ccag',
            'sousTraitanceDeclaree', 'typeGroupementOperateurs', 'tauxAvance',
            'origineFrance', 'idAccordCadre', 'montant', 'marcheInnovant', 'codeCPV']]
        return df
    elif cat == 'anomalie':
        df = df[['Ronan : à remplir avec les colonnes que tu souhaites']]
        return df
    else:
        return f"Error, cat not in 'montant', 'marche_sim', 'anomalie'."


#retrait des marchés supérieurs à 50 millions et inférieur à 20 milles.
def drop_outliers(df, min=20000, max=50000000):
    df.drop(df[df['montant'] > max].index, inplace=True)
    df.drop(df[df['montant'] < min].index, inplace=True)
    df.drop(df[df['dureeMois'] == 999].index, inplace=True)
    return df


#ajout d'une colonne avec les 2 premiers chiffres du CPV, et les trois premiers si c'est 45 ou 71
def cpv_2et3 (df):
    df = add_cpv_hierarchy_column(df)
    df3 = add_cpv_hierarchy_column(df, level=3)
    df["codeCPV_3"] = df3["codeCPV_3"]
    df['codeCPV_2'] = df.apply(lambda row: row['codeCPV_3'] if row['codeCPV_2'] == '45000000' else row['codeCPV_2'], axis=1)
    df['codeCPV_2'] = df.apply(lambda row: row['codeCPV_3'] if row['codeCPV_2'] == '71000000' else row['codeCPV_2'], axis=1)
    df.drop(columns=['codeCPV_3'], inplace=True)
    return df


#créer une colonne 'année' en version datetime
def annee(df):
    df['annee'] = df['dateNotification'].str[:4]
    df['annee'] = pd.to_datetime(df['annee'], errors='ignore')
    df = df[df['annee'] > '2018']
    return df
