#from sklearn.impute import SimpleImputer
#
#def clean_missing_values(df):
#    """
#    Handles missing data in the dataset:
#    1. Drops rows with missing 'dureeMois' values
#    2. Replaces missing values in 'marcheInnovant' with 0
#    3. Replaces missing values in 'tauxAvance' with 0
#
#    Parameters:
#    df (DataFrame): Input dataframe containing the data to clean
#
#    Returns:
#    DataFrame: Cleaned dataframe with missing values handled
#    """
#    df_clean = df.copy()
#
#    #dureeMois
#    df_clean = df_clean.dropna(subset=['dureeMois'])
#
#    # marcheInnovant
#    marcheInnovant_imputer = SimpleImputer(strategy='constant', fill_value=0.0)
#    df_clean['marcheInnovant'] = marcheInnovant_imputer.fit_transform(df_clean[['marcheInnovant']])
#
#    # tauxAvance
#    avance_imputer = SimpleImputer(strategy='constant', fill_value=0.0)
#    df_clean['tauxAvance'] = avance_imputer.fit_transform(df_clean[['tauxAvance']])
#
#    # offresRecues
#    offres_recues_imputer = SimpleImputer(strategy='median')
#    df_clean['offresRecues'] = offres_recues_imputer.fit_transform(df_clean[['offresRecues']])
#
#    # sousTraitanceDeclaree
#    sous_traitance_imputer = SimpleImputer(strategy='constant', fill_value=0.0)
#    df_clean['sousTraitanceDeclaree'] = sous_traitance_imputer.fit_transform(df_clean[['sousTraitanceDeclaree']])
#
#    # origineFrance
#    origine_france_imputer = SimpleImputer(strategy='constant', fill_value=0.0)
#    df_clean['origineFrance'] = origine_france_imputer.fit_transform(df_clean[['origineFrance']])
#
#    return df_clean
#
