import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_categorielle(X, columns):
    """
    Preprocess categorical features using OneHotEncoder.

    Parameters:
    X (DataFrame): Input data containing categorical features.
    columns (list): List of column names to be encoded.

    Returns:
    DataFrame: Transformed data with one-hot encoded categorical features.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(X[columns])

    # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(columns))

    # Drop original categorical columns and concatenate the new encoded columns
    X_transformed = X.drop(columns=columns).reset_index(drop=True)
    X_transformed = pd.concat([X_transformed, encoded_df], axis=1)

    return X_transformed
