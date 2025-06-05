import pandas as pd
from scripts.preprocess_cpv import add_cpv_hierarchy_column

#retrait des marchés supérieurs à 50 millions et inférieur à 20 milles.
def drop_outliers(df, min=20000, max=50000000):
    df.drop(df[df['montant'] > max].index, inplace=True)
    df.drop(df[df['montant'] < min].index, inplace=True)
    df.drop(df[df['dureeMois'] == 999].index, inplace=True)
    return df

#ajout d'une colonne avec les 2 premiers chiffres du CPV, et les trois premiers si c'est 45 ou 71
def cpv_2et3(df):
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


#selection des colonnes
def columns_selection(df, cat, min=20000, max=50000000, top_n=40, level=2):

    if cat == 'pred_montant':
        #selection des colonnes
        df = df[['procedure', 'dureeMois','nature', 'formePrix', 'offresRecues', 'ccag',
            'sousTraitanceDeclaree', 'typeGroupementOperateurs', 'tauxAvance_cat',
            'origineFrance', 'idAccordCadre', 'dateNotification', 'marcheInnovant', 'codeCPV']]
        #ajout de la colonne codeCPV_2
        df = cpv_2et3(df)
        #drop cpv moins representés
        cpv_group_counts = df['codeCPV_2'].value_counts()
        top_n = top_n
        top_groups = cpv_group_counts.nlargest(top_n)
        df = df[df['codeCPV_2'].isin(top_groups.index)]
        #drop colonne codeCPV
        df.drop(columns=['codeCPV'], inplace=True)
        #ajout de la colonne année
        df = annee(df)
        #suppression des outiliers (montant sup, inf et dureeMois sup)
        #df = drop_outliers(df, min=min, max=max)
        return df

    elif cat == 'marche_sim':
        #selection des colonnes
        df = df[['procedure', 'dureeMois','nature', 'formePrix', 'offresRecues', 'ccag',
            'sousTraitanceDeclaree', 'typeGroupementOperateurs', 'tauxAvance_cat',
            'origineFrance', 'idAccordCadre', 'montant', 'marcheInnovant', 'codeCPV']]
        #ajout de la colonne codeCPV_2
        df = add_cpv_hierarchy_column(df, level=level)
        #drop cpv moins representés
        cpv_group_counts = df['codeCPV_2'].value_counts()
        top_n = top_n
        top_groups = cpv_group_counts.nlargest(top_n)
        df = df[df['codeCPV_2'].isin(top_groups.index)]
        #drop colonne codeCPV
        df.drop(columns=['codeCPV'], inplace=True)
        #suppression des outiliers (montant sup, inf et dureeMois sup)
        df = drop_outliers(df, min=min, max=max)
        return df

    elif cat == 'anomalie':
        #selection des colonnes
        df = df[['Ronan : à remplir avec les colonnes que tu souhaites']]
        #ajout de la colonne codeCPV_3
        df = add_cpv_hierarchy_column(df, level=level)
        #drop cpv moins representés ? (à ajouter si oui)
        #ajout de la colonne annee
        df = annee(df)
        #suppression des outiliers (montant sup, inf et dureeMois sup)
        df = drop_outliers(df, min=min, max=max)
        return df

    else:
        return "Error, cat not in 'montant', 'marche_sim', 'anomalie'."
