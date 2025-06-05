import pandas as pd

from scripts.preprocess_cpv import add_cpv_hierarchy_column


def cpv_2et3(df):
    """
    Add column with first 2 digits of CPV, and first 3 if it's 45 or 71.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with modified codeCPV_2 column
    """
    try:
        # Check if codeCPV_2 already exists, if not create it
        if 'codeCPV_2' not in df.columns:
            df = add_cpv_hierarchy_column(df)
        
        # Check if codeCPV_3 already exists, if not create it
        if 'codeCPV_3' not in df.columns:
            df = add_cpv_hierarchy_column(df, level=3)
    except Exception as e:
        print(f"Error in cpv_2et3: {e}")
        return df
        
    df['codeCPV_2'] = df.apply(
                        lambda row: row['codeCPV_3'] if row['codeCPV_2'] in [45000000, 71000000] else row['codeCPV_2'],
                        axis=1
                    )
                        
    # Drop columns that exist
    columns_to_drop = ['codeCPV_3', 'codeCPV_4', 'codeCPV_5']
    existing_columns_to_drop = [col for col in columns_to_drop 
                                if col in df.columns]
    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)
    
    return df


def annee(df):
    """
    Create an 'annee' column in datetime format from dateNotification.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'annee' column and filtered for years > 2018
    """
    try:
        if 'annee' not in df.columns:
            df['annee'] = df['dateNotification'].str[:4]
            df['annee'] = pd.to_datetime(df['annee'], errors='ignore')
            df = df[df['annee'] > '2018']
    except Exception as e:
        print(f"Error in annee: {e}")
        return df
    return df


def create_columns(df):
    """
    Run both cpv_2et3 and annee functions to create all necessary columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with CPV hierarchy and annee columns created
    """
    df = cpv_2et3(df)
    df = annee(df)
    return df


