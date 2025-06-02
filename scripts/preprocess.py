'''
This script is used to add CPV descriptions to the raw data by matching CPV codes.
'''

# Import libraries
import pandas as pd
import os


def codeCPV_description(data_raw):
    """
    Add CPV descriptions to the raw data by matching CPV codes.
    
    Args:
        data_raw (pd.DataFrame): Raw data containing 'codeCPV' column
        
    Returns:
        pd.DataFrame: Data with added 'codeCPV_FR' column containing 
                     CPV descriptions in French
    """
    # Load CPV reference data
    cpv_path = os.path.join(os.path.dirname(os.getcwd()), 
                            'docs', 'cpv_2008_ver_2013_FR.csv')
    df_cpv = pd.read_csv(cpv_path)
    
    # Find missing CPV codes (not in reference data)
    missing_cpv_codes = data_raw[
        ~data_raw['codeCPV'].isin(df_cpv['CODE'])
    ]['codeCPV'].unique()
    missing_cpv = pd.DataFrame(missing_cpv_codes, columns=['codeCPV'])
    
    # Try to find similar CPV codes for missing ones
    missing_cpv.loc[:, 'count_similar'] = (
        missing_cpv['codeCPV'].astype(str).apply(
            lambda cpv: df_cpv['CODE'].str.startswith(cpv).sum()
        )
    )
    
    missing_cpv.loc[:, 'new_CPV'] = (
        missing_cpv['codeCPV'].astype(str).apply(
            lambda cpv: (
                df_cpv[df_cpv['CODE'].str.startswith(cpv)]['CODE'].values[0] 
                if df_cpv[df_cpv['CODE'].str.startswith(cpv)].shape[0] > 0 
                else None
            )
        )
    )
    
    # Merge missing CPV codes with their descriptions
    missing_cpv = pd.merge(
        missing_cpv, df_cpv[['CODE', 'FR']], 
        left_on='new_CPV', right_on='CODE', how='left'
    )
    missing_cpv.rename(columns={'FR': 'codeCPV_FR'}, inplace=True)
    
    # Handle correct CPV codes (already in reference data)
    correct_cpv_codes = data_raw[
        data_raw['codeCPV'].isin(df_cpv['CODE'])
    ]['codeCPV'].unique()
    correct_cpv = pd.DataFrame(correct_cpv_codes, columns=['codeCPV'])
    
    correct_cpv = pd.merge(
        correct_cpv, df_cpv[['CODE', 'FR']], 
        left_on='codeCPV', right_on='CODE', how='left'
    )
    correct_cpv.rename(columns={'FR': 'codeCPV_FR'}, inplace=True)
    
    # Combine both correct and missing CPV mappings
    cpvFR = pd.concat([
        correct_cpv[['codeCPV', 'codeCPV_FR']], 
        missing_cpv[['codeCPV', 'codeCPV_FR']]
    ])
    
    # Merge original data with CPV descriptions
    data_cpv = pd.merge(
        data_raw, cpvFR, left_on='codeCPV', 
        right_on='codeCPV', how='left'
    )
    
    return data_cpv


