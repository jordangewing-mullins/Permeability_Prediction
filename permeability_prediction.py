import argparse
from rdkit import Chem
from rdkit.Chem import Descriptors
import pickle
import numpy as np
from custom_functions import *
from Exploratory_data_analysis import *
import joblib

def load_scaler(scaler_file):
    """
    Load a scaler from a saved file.

    Parameters:
        scaler_file (str): The path to the saved scaler file.

    Returns:
        scaler: The loaded scaler object.
    """
    with open(scaler_file, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return scaler

def canonicalize_dataframe(df: pd.DataFrame):
    """
    Canonicalize a DataFrame by alphabetically ordering its column names.

    The goal of this function is to put all the column names in alphabetical order so that
    we can avoid an additional hard coding step when working with the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be canonicalized.

    Returns:
        pd.DataFrame: A new DataFrame with column names sorted alphabetically.
    """
    existing_columns = df.columns
    sorted_columns = sorted(existing_columns)
    # Create a mapping dictionary to rename columns alphabetically
    rename_dict = {existing_columns[i]: sorted_columns[i] for i in range(len(existing_columns))}
    canon_df = df.rename(columns=rename_dict)
    return canon_df

def predict_perm_from_smiles(smiles, scaler, model):
    """
    Predict permeability from SMILES notation using a pre-trained scaler and model.

    Given a SMILES notation, this function calculates relevant features, scales them using a pre-trained scaler,
    and then uses a pre-trained machine learning model to predict the permeability.

    Parameters:
        smiles (str or list of str): The SMILES notation(s) for which permeability is to be predicted.
        scaler: The pre-trained scaler used for feature scaling.
        model: The pre-trained machine learning model used for prediction.

    Returns:
        np.ndarray: Predicted permeability values corresponding to the input SMILES.
    """
    df = pd.DataFrame({'canonical_smiles':smiles}, index=[0])
    featurized_data = add_features(df)
    features_to_transform = ['num_rotatable_bonds','num_hydrogen_bond_donors','num_hydrogen_bond_acceptors','num_rings','num_aromatic_rings','num_het_atoms','MaxPartialCharge','tpsa','mol_weight']
    new_feature_names = [x + '_Transformed' for x in features_to_transform]

    # Get it in the right format for scaling (need to change column names) -- ideally this wouldn't be hard-coded
    featurized_data = featurized_data.rename(columns=dict(zip(features_to_transform, new_feature_names)))
    featurized_data = featurized_data.drop(columns=['canonical_smiles','fingerprint'])

    # Get all the columns in the same order as the training data
    featurized_data = canonicalize_dataframe(featurized_data)
    features_scaled = scaler.transform(featurized_data) 
    prediction = model.predict(features_scaled)
    un_logged_prediction = np.exp(prediction)
    return un_logged_prediction

# Retrieve the best model and cognate scaler
scaler_file = 'LASSO_scaler.pkl'
model = joblib.load("lasso_model_permeability.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict permeability from SMILES string.")
    parser.add_argument("smiles", type=str, help="Input SMILES string")
    args = parser.parse_args()

    # Retrieve the best model and cognate scaler
    scaler_file = 'LASSO_scaler.pkl'
    model = joblib.load("lasso_model_permeability.pkl")

    prediction = predict_perm_from_smiles(args.smiles, load_scaler(scaler_file), model)
    print(f"Predicted permeability for SMILES '{args.smiles}': {prediction}")