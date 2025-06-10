import os
import pickle
from scripts.preprocess_pipeline import create_pipeline

data_path = os.path.join(os.path.dirname(__file__), '..', 'data')


def save_montant_prediction_pipeline():
    """Create and save the pipeline for montant prediction."""
    # Estimation du montant
    numerical_columns = ['dureeMois', 'offresRecues', 'annee']

    binary_columns = ['sousTraitanceDeclaree', 'origineFrance',
                      'marcheInnovant', 'idAccordCadre']

    categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                           'typeGroupementOperateurs', 'tauxAvance_cat',
                           'codeCPV_3', 'acheteur_tranche_effectif',
                           'acheteur_categorie']

    pipeline_pred_montant = create_pipeline(numerical_columns,
                                            binary_columns,
                                            categorical_columns)

    # Save the pipeline to a file
    with open(os.path.join(data_path, 'pipeline_pred_montant.pkl'),
              'wb') as f:
        pickle.dump(pipeline_pred_montant, f)
    
    print("Pipeline for montant prediction saved successfully.")


def save_marche_similaire_pipeline():
    """Create and save the pipeline for similar marches."""
    # Marchés similaires
    numerical_columns = ['montant', 'dureeMois', 'offresRecues']
    binary_columns = ['sousTraitanceDeclaree', 'origineFrance',
                      'marcheInnovant', 'idAccordCadre']
    categorical_columns = ['procedure', 'nature', 'formePrix', 'ccag',
                           'typeGroupementOperateurs', 'tauxAvance_cat',
                           'codeCPV_2_3']

    pipeline_marche_sim = create_pipeline(numerical_columns,
                                          binary_columns,
                                          categorical_columns)

    # Save the pipeline to a file
    with open(os.path.join(data_path, 'pipeline_marche_sim.pkl'),
              'wb') as f:
        pickle.dump(pipeline_marche_sim, f)
    
    print("Pipeline for similar marches saved successfully.")


if __name__ == "__main__":
    print("Creating and saving pipelines...")
    save_montant_prediction_pipeline()
    save_marche_similaire_pipeline()
    print("All pipelines saved successfully!")

