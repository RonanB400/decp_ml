import pandas as pd
from sklearn.preprocessing import OneHotEncoder



def encoding_idAccordCadre(X):
    """
    Encodes the idAccordCadre column by replacing non-null values with 1 and null values with 0.

    Parameters:
    X (DataFrame): Input data containing the idAccordCadre column.

    Returns:
    DataFrame: A copy of the input DataFrame with the idAccordCadre column encoded.
    """
    X_encoded = X.copy()

    X_encoded['idAccordCadre'] = X_encoded['idAccordCadre'].notnull().astype(int)

    return X_encoded



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


    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(columns))

    X_transformed = X.drop(columns=columns).reset_index(drop=True)
    X_transformed = pd.concat([X_transformed, encoded_df], axis=1)

    return X_transformed
