

import pandas as pd
import numpy as np
import os


data_path = os.path.join(os.path.dirname(os.getcwd()), 'data')
db_path = os.path.join(data_path, 'datalab.sqlite')

data_cpv = pd.read_csv(os.path.join(data_path, 'data_cpv.csv'))

def extract_cpv_hierarchy_level(cpv_code, level=2):
    """
    Extract higher-level hierarchy code from a CPV code.
    
    Args:
        cpv_code (str): Original CPV code (e.g., '03111900-1')
        level (str): Hierarchy level to extract. Options:
                    - 'division' (XX000000): First 2 digits + 6 zeros (default)
                    - 'group' (XXXX0000): First 4 digits + 4 zeros  
                    - 'class' (XXXXXX00): First 6 digits + 2 zeros
    
    Returns:
        str: Higher-level CPV code (e.g., '03000000')
    """
    # Remove any whitespace and convert to string
    cpv_str = str(cpv_code).strip()
    
    # Extract the numeric part before the dash
    if '-' in cpv_str:
        numeric_part = cpv_str.split('-')[0]
    else:
        numeric_part = cpv_str
    
    # Ensure we have at least 8 digits, pad with zeros if needed
    numeric_part = numeric_part.ljust(8, '0')
    
    # Extract based on hierarchy level
    if level == 2:
        # First 2 digits + 6 zeros (e.g., 03111900 -> 03000000)
        return numeric_part[:2] + '000000'
    elif level == 3:
        # First 3 digits + 5 zeros (e.g., 03111900 -> 03111000)
        return numeric_part[:3] + '00000'
    elif level == 4:
        # First 4 digits + 4 zeros (e.g., 03111900 -> 03110000)
        return numeric_part[:4] + '0000'
    elif level == 5:
        # First 5 digits + 3 zeros (e.g., 03111900 -> 03111000)
        return numeric_part[:5] + '000'
    else:
        raise ValueError("Level must be between 2 and 5")


def add_cpv_hierarchy_column(df, cpv_column='codeCPV', level=2, 
                             new_column_name=None):
    """
    Add a new column with higher-level CPV hierarchy codes to a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing CPV codes
        cpv_column (str): Name of the column containing CPV codes 
                         (default: 'codeCPV')
        level (str): Hierarchy level to extract (default: 'division')
        new_column_name (str): Name for the new column. If None, 
                              will be auto-generated.
    
    Returns:
        pd.DataFrame: DataFrame with added hierarchy column
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Auto-generate column name if not provided
    if new_column_name is None:
        new_column_name = f'codeCPV_{level}'
    
    # Apply the hierarchy extraction function
    df_copy[new_column_name] = df_copy[cpv_column].apply(
        lambda x: extract_cpv_hierarchy_level(x, level=level)
    )
    
    return df_copy


data_cpv_new = data_cpv.copy()
for i in range(2, 6):
    data_cpv_new = add_cpv_hierarchy_column(data_cpv_new, level=i)
    

data_cpv_new.to_csv(os.path.join(data_path, 'data_cpv.csv'), index=True)

return data_cpv_new



