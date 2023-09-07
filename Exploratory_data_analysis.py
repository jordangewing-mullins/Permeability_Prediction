# Exploratory data visualizations
import matplotlib.pyplot as plt
from custom_functions import * 
from scipy import stats

# Import the file
file_path = 'perm.tsv'
df = pd.read_csv(file_path, delimiter='\t')

def add_features(df: pd.DataFrame):
    """In this fucntion we can apply all of the relevant RDKit descriptors to each molecule, returning a 
    dataframe with more features than can be used to build a predictive model
    
    Parameters:
    - df (DataFrame): The input DataFrame containing the raw data, as well as the target variable.

    Returns:
    - df (DataFrame): An updated DataFrame that includes 2000+ features off of which we can train our models

    """
    descriptor_columns = df['canonical_smiles'].apply(calculate_descriptors)
    df = pd.concat([df, descriptor_columns], axis=1)
    return df

def initial_vis(featurized_data: pd.DataFrame):
    """
    Perform initial visualization of relevant predictor variables.

    This function generates histograms for a selection of relevant predictor variables
    in the provided DataFrame, allowing for an initial exploration of their distributions.

    Args:
        featurized_data (pd.DataFrame): The DataFrame containing the featurized predictor variables.

    Returns:
        None
    """
    # Now let's take a look at the predictors that aren't fingerprints
    relevant_predictors = ['num_rotatable_bonds',
        'num_hydrogen_bond_donors',
        'num_hydrogen_bond_acceptors',
        'logp',
        'num_rings',
        'num_aromatic_rings',
        'num_het_atoms',
        'MaxPartialCharge','tpsa','mol_weight']

    predictors_df_for_viz = featurized_data[relevant_predictors]
    fig, axs = plt.subplots(2, 5, figsize=(16, 8))
    fig.suptitle('Histograms for Predictor Variables')

    # Plot histograms for each predictor
    for i, column in enumerate(predictors_df_for_viz.columns):
        row, col = divmod(i, 5)
        axs[row, col].hist(predictors_df_for_viz[column], bins=20, edgecolor='black', alpha=0.7)
        axs[row, col].set_title(column)
        axs[row, col].set_xlabel('Value')
        axs[row, col].set_ylabel('Frequency')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_name = 'Histograms_for_Predictor_Variables.png'
    plt.savefig(fig_name)
    print(f"Saving figure as {fig_name}")
    return None

def boxcox_transform_dataframe(input_df: pd.DataFrame, column_names: list, epsilon=0.01):
    """
    Apply Box-Cox transformation to specified columns in a DataFrame.

    Args:
        input_df (pd.DataFrame): The input DataFrame.
        column_names (list of str): A list of column names to be transformed.
        epsilon (float, optional): A small constant to add to the data to ensure positivity. Default is 0.01.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns transformed using Box-Cox.
    """

    df_transformed = input_df.copy()

    if not isinstance(column_names, list):
        column_names = [column_names]

    for column_name in column_names:
        # Check if the specified column exists in the DataFrame
        if column_name not in df_transformed.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        # Add epsilon to the specified column to ensure positivity
        df_transformed[column_name] += epsilon

        # Apply the Box-Cox transformation to the specified column
        transformed_data, lambda_value = stats.boxcox(df_transformed[column_name])

        # Create a new column with the transformed data
        df_transformed[column_name + '_Transformed'] = transformed_data

    return df_transformed

def plot_transformed_predictors(transformed_df: pd.DataFrame):
    """
    Plot histograms for transformed predictor variables.

    This function generates histograms for each transformed predictor variable in the provided DataFrame.

    Args:
        transformed_df (pd.DataFrame): The DataFrame containing the transformed predictor variables.

    Returns:
        None
    """
    # Get the number of columns in the DataFrame
    num_columns = len(transformed_df.columns)
    
    # Calculate the number of rows and columns for the subplots
    num_rows = (num_columns - 1) // 5 + 1
    num_cols = min(num_columns, 5)
    
    # Create subplots with the calculated layout
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 8))
    fig.suptitle('Histograms for Transformed Predictor Variables')

    # Flatten the axs array if it's multi-dimensional
    axs = axs.ravel()
    
    for i, column in enumerate(transformed_df.columns):
        axs[i].hist(transformed_df[column], bins=20, edgecolor='black', alpha=0.7)
        axs[i].set_title(column)
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
    
    # Remove any remaining empty subplots
    for i in range(num_columns, num_rows * num_cols):
        fig.delaxes(axs[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_name = 'Histograms_for_Transformed_Predictor_Variables.png'
    plt.savefig(fig_name)
    print(f"Saving figure as {fig_name}")
    return None

def visualize_response(transformed_df: pd.DataFrame):
    """
    Visualize the response variable and identify potential outliers.

    This function plots the response variable 'Papp_in_cm_sec' to visualize its distribution
    and identify potential outliers.

    Args:
        transformed_df (pd.DataFrame): The DataFrame containing the transformed data.

    Returns:
        None
    """
    repsonse_var = transformed_df['Papp_in_cm_sec']

    # Let's take a look at the response variable, Papp_in_cm_sec
    plt.figure(figsize=(8, 6))
    plt.plot(repsonse_var, 'bo', markersize=3) 
    plt.xlabel('Data Index')
    plt.ylabel('Response Variable (Papp_in_cm_sec)')
    plt.title('Dot Plot of Response Variable')
    plt.grid(True)
    plt.tight_layout()
    fig_name = 'permeability_raw_data.png'
    plt.savefig(fig_name)
    print(f"Saving figure as {fig_name}")
    # From these data we can see that there are a few outliers on the upper end, so I will remove any reponse data point with |Z|>3.
    return None


def remove_outliers(df: pd.DataFrame, column_name: str, z_threshold=3):
    """
    Remove outliers from a DataFrame based on Z-scores.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column in the DataFrame for which outliers should be removed.
        z_threshold (float, optional): The Z-score threshold for removing outliers. Data points with absolute
            Z-scores greater than this threshold will be considered outliers. Default is 3.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed based on the specified Z-score threshold.
    """
    z_scores = stats.zscore(df[column_name])
    mask = abs(z_scores) <= z_threshold
    filtered_df = df[mask]
    number_of_outliers = len(df)-len(filtered_df)
    print(f"{number_of_outliers} outliers were removed. This leads to a {round(100*number_of_outliers/len(df),2)}% reduction in the amount of data")

    return filtered_df

def canonicalize_dataframe(df: pd.DataFrame):
    # The goal of this function is to put all the column names in alphabetical order so that we can avoid an additional hard coding step 
    existing_columns = df.columns
    sorted_columns = sorted(existing_columns)
    # Create a mapping dictionary to rename columns alphabetically
    rename_dict = {existing_columns[i]: sorted_columns[i] for i in range(len(existing_columns))}
    canon_df = df.rename(columns=rename_dict)
    return canon_df

if __name__ == "__main__":
    featurized_data = add_features(df) # Add features to the data
    initial_vis(featurized_data) # Visualize raw data
    # From these plots we see that a lot of the data aren't normal, which could be problematic for building a model
    # So let's do a Box-Cox transformation to potentially ameliorate this issue!
    features_to_transform = ['num_rotatable_bonds','num_hydrogen_bond_donors','num_hydrogen_bond_acceptors','num_rings','num_aromatic_rings','num_het_atoms','MaxPartialCharge','tpsa','mol_weight']
    transformed_data = boxcox_transform_dataframe(featurized_data,features_to_transform) # Transform the data

    plot_transformed_predictors(transformed_data[features_to_transform]) # Visualize transformed data
    # These data look normal now, so we can go ahead and feed them into our model. But first we need to add back in the response variable.
    transformed_data['Papp_in_cm_sec'] = featurized_data['Papp_in_cm_sec'].copy()
    visualize_response(transformed_data) 
    # Now filter out outliers
    filtered_df = remove_outliers(transformed_data, 'Papp_in_cm_sec').copy()
    # The final act of data clean up I'll do is applying a logarithmic transformation to the reponse variable. This is for 2 reasons. 1 being that the values are super small (on the order of 1e-10)
    # so I think I can resolve predictions better by doing the log of permeability. The next reason is because I can make the data more symmetrical and closer to a normal distribution, 
    # which should be beneficial because I want to build some statistical models that assume normally distributed residuals.
    filtered_df['log_Papp_in_cm_sec'] = np.log(filtered_df['Papp_in_cm_sec'])
    #filtered_df = filtered_df.loc[:, filtered_df.columns != 'Papp_in_cm_sec']
    columns_to_remove = features_to_transform + ['molregno', 'canonical_smiles','Papp_in_cm_sec','fingerprint']
    # Remove the specified columns from filtered_df
    filtered_df = filtered_df.drop(columns=columns_to_remove)
    # Canonicalize the dataframe
    df_to_feed_into_models = canonicalize_dataframe(filtered_df.copy()) 
    # I wanted to pass this variable directly to the build_model.py file, but I was having issues with that.
    # so I decided to just save the data into a new CSV and open it when building the model.
    df_to_feed_into_models.to_csv('cleaned_up_and_featurized_data.csv')