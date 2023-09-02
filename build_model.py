import tensorflow as tf
import matplotlib.pyplot as plt
from custom_functions import * 

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

def visualize_featurized_data(df: pd.DataFrame):
    return

def transform_features(df: pd.DataFrame):
    """This function logs right skewed data and returns the new corresponding dataframe.

    Parameters:
    - df (DataFrame): The input DataFrame containing relevant and other predictors, as well as the target variable.
    
    Returns:
    - df_new (DataFrame): A new DataFrame with tranformed values
    """

    df['tpsa_transformed'] = np.log1p(df['tpsa'])
    df['mol_weight_transformed'] = np.log1p(df['mol_weight'])

    df_new = df.copy()
    df_new['log_Papp_in_cm_sec'] = np.log(df['Papp_in_cm_sec'])
    columns_to_remove = ['Papp_in_cm_sec','mol_weight', 'tpsa','fingerprint', 'canonical_smiles','molregno']
    df_new = df_new.drop(columns=columns_to_remove)
    return df_new

# Make the dataframe to feed into each model
df_to_feed_into_models = (transform_features(add_features(df)))

def build_lasso_regression_model(df_to_feed_into_models: pd.DataFrame):
    predictors = [col for col in df_to_feed_into_models.columns if col not in ['log_Papp_in_cm_sec']]
    X = df_to_feed_into_models[predictors]
    y = df_to_feed_into_models['log_Papp_in_cm_sec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a range of alpha values to test
    alphas = np.logspace(-6, 2, 100)
    # Initialize LassoCV with 5-fold cross-validation
    lasso_cv = LassoCV(alphas=alphas, cv=5, verbose=False)  
    # Fit LassoCV to the scaled training data
    lasso_cv.fit(X_train_scaled, y_train)
    # Get the selected alpha
    best_alpha = lasso_cv.alpha_
    # Fit the Lasso model with the best alpha on the full training data
    lasso_model = Lasso(alpha=best_alpha)
    lasso_model.fit(X_train_scaled, y_train)

    # Get the selected features
    coef = lasso_model.coef_
    selected_features = [predictors[i] for i in range(len(predictors)) if coef[i] != 0]
    print("Selected features:", selected_features)

    # Evaluate the model's performance on the test data
    y_pred = lasso_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = lasso_model.score(X_test_scaled, y_test)

    # Initialize the Gradient Boosting Regressor
    gradient_boosting_model = GradientBoostingRegressor(
        n_estimators=100,  
        learning_rate=0.1,  
        max_depth=3,       
        random_state=84    
    )

    # Fit the Gradient Boosting model to the scaled training data
    gradient_boosting_model.fit(X_train_scaled, y_train)

    # Predict on the test data using the Gradient Boosting model
    y_pred_gb = gradient_boosting_model.predict(X_test_scaled)

    # Evaluate the Gradient Boosting model's performance
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    r_squared_gb = gradient_boosting_model.score(X_test_scaled, y_test)

    return (r_squared_gb, mse_gb)


def build_neural_net(df_to_feed_into_models: pd.DataFrame):
    """
    Build and train a neural network model.

    Parameters:
    - df_new (DataFrame): The input DataFrame containing relevant and other predictors, as well as the target variable.

    Returns:
    - Tuple[float, float]: A tuple containing the R-squared score and mean squared error of the trained neural network model.
    """

    # Here I'm making a subset of predictors that exclude all the chemical fingerprints 
    relevant_predictors = ['num_rotatable_bonds',
    'num_hydrogen_bond_donors',
    'num_hydrogen_bond_acceptors',
    'logp',
    'num_rings',
    'num_aromatic_rings',
    'num_het_atoms',
    'MaxPartialCharge']

    # Here I'm storing all the predictors that are chemical fingerprints
    other_predictors = [col for col in df_to_feed_into_models.columns if col not in relevant_predictors]

    X_relevant = df_to_feed_into_models[relevant_predictors]
    X_other = df_to_feed_into_models[other_predictors]

    # Split the data into training (70%) and test (30%) sets
    X_train_relevant, X_test_relevant, X_train_other, X_test_other, y_train, y_test = train_test_split(
        X_relevant, X_other, df_to_feed_into_models['log_Papp_in_cm_sec'], test_size=0.3, random_state=84
    )

    # Train a Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)  #  train the gradient boosting model just on the non chemical fingerprint data and make predictions which we can then feed into the neural network
    gb_regressor.fit(X_train_relevant, y_train)

    # Get gradient boosting predictions
    gb_predictions_train = gb_regressor.predict(X_train_relevant)
    gb_predictions_test = gb_regressor.predict(X_test_relevant)

    # Combine gradient boosting predictions with original features
    X_train_combined = np.hstack((X_train_other, gb_predictions_train.reshape(-1, 1)))
    X_test_combined = np.hstack((X_test_other, gb_predictions_test.reshape(-1, 1)))

    # Build a neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_combined.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Convert data to TensorFlow tensors
    X_train_combined_tf = tf.convert_to_tensor(X_train_combined, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test_combined_tf = tf.convert_to_tensor(X_test_combined, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Train the neural network
    model.fit(X_train_combined_tf, y_train_tf, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

    # Evaluate the model on the test data
    mse_nn = model.evaluate(X_test_combined_tf, y_test_tf)
    print(f"Mean Squared Error: {mse_nn}")

    y_pred_nn = model.predict(X_test_combined_tf)
    r_squared_nn = r2_score(y_test, y_pred_nn)
    print(f"R squared: {r_squared_nn}")
    return (r_squared_nn, mse_nn)


# Build all the relevant models and store their accuracy metrics
all_model_metrics_dict = {"Neural Net": build_neural_net(df_to_feed_into_models), "LASSO regression": build_lasso_regression_model(df_to_feed_into_models)}
all_model_metrics = pd.DataFrame.from_dict(all_model_metrics_dict, orient='index', columns=['R-squared', 'MSE'])
all_model_metrics.reset_index(inplace=True)
all_model_metrics.rename(columns={'index': 'Model Type'}, inplace=True)
all_model_metrics = all_model_metrics.sort_values(by='R-squared', ascending=False)
print(all_model_metrics)


# Based on these data it appears that the neural network is the best model for predicting permeability of compounds!
