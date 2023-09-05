import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, Lasso
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        descriptors = {
            'mol_weight': Descriptors.MolWt(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_hydrogen_bond_donors': Descriptors.NumHDonors(mol),
            'num_hydrogen_bond_acceptors': Descriptors.NumHAcceptors(mol),
            'logp': Descriptors.MolLogP(mol),
            'num_rings': Descriptors.RingCount(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_het_atoms': Descriptors.NumHeteroatoms(mol),
            'MaxPartialCharge':Descriptors.MaxPartialCharge(mol, force=False),
            'fingerprint':list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        }
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fingerprint_array = np.array(list(fingerprint), dtype=int)  # Convert the fingerprint to a numpy array
        for i, fingerprint_value in enumerate(fingerprint_array):
            descriptors[f'fingerprint_{i}'] = fingerprint_value

        return pd.Series(descriptors)
    else:
        return pd.Series([None] * 11)  # Return None for missing or invalid SMILES