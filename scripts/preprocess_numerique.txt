import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def log_transformation(X, columns):
    """
    Apply logarithmic transformation to specified columns in the DataFrame.

    Parameters:
    X (DataFrame): Input data containing the columns to be transformed.
    columns (list): List of column names to apply the log transformation.

    Returns:
    DataFrame: A copy of the input DataFrame with specified columns transformed
    """
    X_transformed = X.copy()
    for column in columns:
        # add 1 to avoid log(0)
        X_transformed[column] = X_transformed[column].apply(lambda x: np.log(x + 1) if x > 0 else 0)
    return X_transformed



def scale_numerical_features(df, columns_numerical):
    """
    Apply StandardScaler to numerical features in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with log-transformed numerical features
    columns_numerical : list
        List of numerical column names

    Returns:
    --------
    pandas.DataFrame, StandardScaler
        Scaled DataFrame and the fitted scaler
    """
    df_scaled = df.copy()

    scaler = StandardScaler()
    df_scaled[columns_numerical] = pd.DataFrame(
        scaler.fit_transform(df[columns_numerical]),
        columns=columns_numerical,
        index=df.index
    )

    return df_scaled
