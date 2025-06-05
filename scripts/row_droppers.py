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


def filter_top_cpv_categories(df, top_n=40, cpv_column='codeCPV_2'):
    """
    Filter DataFrame to keep only rows with the top N most frequent CPV categories.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame
    top_n : int, default=40
        Number of top CPV categories to keep
    cpv_column : str, default='codeCPV_2'
        Name of the CPV column to filter on
        
    Returns
    -------
    pandas.DataFrame
        DataFrame filtered to top N CPV categories
    """
    try:
        # Check if CPV column exists
        if cpv_column not in df.columns:
            print(f"Warning: Column '{cpv_column}' not found in DataFrame")
            return df
        
        # Count occurrences of each CPV category
        cpv_group_counts = df[cpv_column].value_counts()
        
        # Get the top N categories
        top_groups = cpv_group_counts.nlargest(top_n)
        
        # Filter DataFrame to keep only top categories
        df_filtered = df[df[cpv_column].isin(top_groups.index)]
        
        print(f"Filtered from {len(cpv_group_counts)} to {len(top_groups)} "
              f"CPV categories, keeping {len(df_filtered)} rows out of {len(df)}")
        
        return df_filtered
        
    except Exception as e:
        print(f"Error in filter_top_cpv_categories: {e}")
        return df

