import pandas as pd


def drop_outliers(df, min=20000, max=50000000):
    """
    Remove rows with outlier values in montant and dureeMois columns.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    min : int, default=20000
        Minimum value threshold for montant
    max : int, default=50000000
        Maximum value threshold for montant
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    try:
        # Check if 'montant' column exists before filtering
        if 'montant' in df.columns:
            df = df.drop(df[df['montant'] > max].index)
            df = df.drop(df[df['montant'] < min].index)
        
        # Check if 'dureeMois' column exists before filtering
        if 'dureeMois' in df.columns:
            df = df.drop(df[df['dureeMois'] > 900].index)
            df = df.dropna(subset=['dureeMois'])
            
    except Exception as e:
        print(f"Error in drop_outliers: {e}")
        return df
    return df

