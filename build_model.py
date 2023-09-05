import tensorflow as tf
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from custom_functions import * 
from scipy.stats import randint     
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pickle

# Read in the data from the CSV generated in Exploratory_data_analysis.py
df_to_feed_into_models = pd.read_csv('cleaned_up_and_featurized_data.csv',index_col=0)

def build_lasso_regression_model(df_to_feed_into_models: pd.DataFrame):
    """
    Build and train a Lasso regression model with feature selection and hyperparameter tuning.

    Parameters:
    - df_to_feed_into_models (pd.DataFrame): The input DataFrame containing predictors and target variable.

    Returns:
    - Tuple[object, float, float]: A tuple containing the model, R-squared and Mean Squared Error (MSE).

    This function takes a DataFrame as input, extracts the predictors and target variable,
    performs data preprocessing including standardization, conducts hyperparameter tuning using LassoCV,
    fits a Lasso regression model, and returns the model along with evaluation metrics.

    Example usage:
    lasso_model, r_squared, mse = build_lasso_regression_model(df)
    """
    predictors = [col for col in df_to_feed_into_models.columns if col not in ['log_Papp_in_cm_sec']]
    X = df_to_feed_into_models[predictors]
    y = df_to_feed_into_models['log_Papp_in_cm_sec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Create a range of alpha values to test
    alphas = np.logspace(-6, 2, 100)
    # Initialize LassoCV with 4-fold cross-validation
    lasso_cv = LassoCV(alphas=alphas, cv=4, verbose=False)  
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

    # Evaluate the model's performance on the test data
    y_pred = lasso_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)
    return (lasso_model, r_squared, mse)


def build_random_forest_model(df_to_feed_into_models: pd.DataFrame, interactions=False):
    """
    Build and train a Random Forest regression model with hyperparameter tuning.

    Parameters:
    - df_to_feed_into_models (pd.DataFrame): The input DataFrame containing predictors and target variable.
    - interactions (bool, optional): If True, consider interaction terms between non-fingerprint predictors.

    Returns:
    - Tuple[object, float, float]: A tuple containing the best model,  R-squared and Mean Squared Error (MSE) achieved
      by the Random Forest model.

    This function builds and trains a Random Forest regression model with optional interaction term consideration.
    It performs hyperparameter tuning using RandomizedSearchCV and returns the best model along with evaluation metrics.

    Example usage:
    best_model, r_squared, mse = build_random_forest_model(df)
    """
    predictors = [col for col in df_to_feed_into_models.columns if col not in ['log_Papp_in_cm_sec']]
    X = df_to_feed_into_models[predictors]
    y = df_to_feed_into_models['log_Papp_in_cm_sec']

    if not interactions:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)
    
    else:
        # Here I'm making a subset of predictors that exclude all the chemical fingerprints 
        relevant_predictors = ['num_rotatable_bonds_Transformed',
                               'num_hydrogen_bond_donors_Transformed',
                               'num_hydrogen_bond_acceptors_Transformed',
                               'logp',
                               'num_rings_Transformed',
                               'num_aromatic_rings_Transformed',
                               'num_het_atoms_Transformed',
                               'MaxPartialCharge_Transformed',
                               'tpsa_Transformed',
                               'mol_weight_Transformed']
        X_relevant = df_to_feed_into_models[relevant_predictors]

        # Initialize the PolynomialFeatures transformer
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_interactions = poly.fit_transform(X_relevant)
        
        # Manually create feature names for interaction terms
        interaction_feature_names = list(relevant_predictors)  # Create a copy of relevant_predictors
        for i, feature1 in enumerate(relevant_predictors):
            for j, feature2 in enumerate(relevant_predictors):
                if i < j:
                    interaction_feature_names.append(f"{feature1}_x_{feature2}")
        
        # Create a DataFrame from the transformed features
        df_interactions = pd.DataFrame(X_interactions, columns=interaction_feature_names)
    
        only_the_interactions = df_interactions.drop(columns=relevant_predictors)

        # Combine the original predictors with the interaction features
        df_combined = pd.concat([X, only_the_interactions], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(df_combined, y, test_size=0.3, random_state=84)

    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2'],
    }

    rf = RandomForestRegressor()
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=100,  # Number of random combinations to try
        scoring='neg_mean_squared_error',  # Use an appropriate scoring metric
        cv=5,  # Number of cross-validation folds
        n_jobs=-1,  # Use all available CPU cores
        random_state=84,  # Set a random seed for reproducibility
        verbose=2,  # Increase verbosity for progress updates
    )
    random_search.fit(X_train_scaled, y_train)
    best_params = random_search.best_params_

    # Iterative hyperparameter tuning - Define a narrowed space around the best parameters
    narrowed_param_dist = {
        'n_estimators': [best_params['n_estimators'] - 3, best_params['n_estimators'], best_params['n_estimators'] + 3],
        'max_depth': [best_params['max_depth'] - 3, best_params['max_depth'], best_params['max_depth'] + 3],
        'min_samples_split': [best_params['min_samples_split'] - 3, best_params['min_samples_split'], best_params['min_samples_split'] + 3],
        'min_samples_leaf': [best_params['min_samples_leaf'] - 3, best_params['min_samples_leaf'], best_params['min_samples_leaf'] + 3],
        'max_features': [best_params['max_features']],  # Keep the best value fixed
    }

    random_search_narrowed = RandomizedSearchCV(rf,param_distributions=narrowed_param_dist,n_iter=30,  # Adjust the number of iterations as needed
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    random_state=84,
    verbose=1)
    random_search_narrowed.fit(X_train_scaled, y_train)
    best_params = random_search_narrowed.best_params_
    

    best_estimator = random_search_narrowed.best_estimator_
    y_pred = best_estimator.predict(X_test_scaled)
    r_squared_rf = r2_score(y_test, y_pred)
    mse_rf = mean_squared_error(y_test, y_pred)

    return (best_estimator, r_squared_rf, mse_rf )


def ensemble_gradient_plus_random_forest(df_to_feed_into_models: pd.DataFrame, interactions=False):
    """
    Build and train an ensemble model combining Gradient Boosting and Random Forest regressors.

    Parameters:
    - df_to_feed_into_models (pd.DataFrame): The input DataFrame containing predictors and target variable.
    - interactions (bool, optional): If True, consider interaction terms between non-fingerprint predictors.

    Returns:
    - Tuple[object, float, float]: A tuple containing the ensemble model, R-squared, and Mean Squared Error (MSE).

    This function builds and trains an ensemble model that combines Gradient Boosting and Random Forest regressors.
    It can consider interaction terms if specified and returns the ensemble model along with evaluation metrics.

    Example usage:
    ensemble_model, r_squared, mse = ensemble_gradient_plus_random_forest(df)
    """
    predictors = [col for col in df_to_feed_into_models.columns if col not in ['log_Papp_in_cm_sec']]
    X = df_to_feed_into_models[predictors]
    y = df_to_feed_into_models['log_Papp_in_cm_sec']

    # These two if statements make different dataframes to feed into the ensemble model based on whether or not we want to explore interaction terms
    if not interactions:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)
        scaler_filename = 'scaler_ensemble_no_interactions.pkl'

    elif interactions:
        print("examining interactions between non-fingerprint terms")
         # Here I'm making a subset of predictors that exclude all the chemical fingerprints 
        relevant_predictors = ['num_rotatable_bonds_Transformed',
                               'num_hydrogen_bond_donors_Transformed',
                               'num_hydrogen_bond_acceptors_Transformed',
                               'logp',
                               'num_rings_Transformed',
                               'num_aromatic_rings_Transformed',
                               'num_het_atoms_Transformed',
                               'MaxPartialCharge_Transformed',
                               'tpsa_Transformed',
                               'mol_weight_Transformed']
        X_relevant = df_to_feed_into_models[relevant_predictors]

        # Initialize the PolynomialFeatures transformer
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_interactions = poly.fit_transform(X_relevant)
        
        # Manually create feature names for interaction terms
        interaction_feature_names = list(relevant_predictors)  # Create a copy of relevant_predictors
        for i, feature1 in enumerate(relevant_predictors):
            for j, feature2 in enumerate(relevant_predictors):
                if i < j:
                    interaction_feature_names.append(f"{feature1}_x_{feature2}")
        
         # Create a DataFrame from the transformed features
        df_interactions = pd.DataFrame(X_interactions, columns=interaction_feature_names)
        
        only_the_interactions = df_interactions.drop(columns=relevant_predictors)

        # Combine the original predictors with the interaction features
        df_combined = pd.concat([X, only_the_interactions], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(df_combined, y, test_size=0.3, random_state=84)
        scaler_filename = 'scaler_ensemble_interactions.pkl'

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler to a file:
    with open(scaler_filename, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    
    # Train the Gradient Boosting model
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=84)
    gb_model.fit(X_train_scaled, y_train)

    # Make predictions using both models
    rf_model = build_random_forest_model(df_to_feed_into_models, interactions=interactions)[0]
    rf_predictions = rf_model.predict(X_test_scaled)  # Random Forest predictions
    gb_predictions = gb_model.predict(X_test_scaled)       # Gradient Boosting predictions

    # Create an ensemble model
    ensemble_X = np.column_stack((rf_predictions, gb_predictions))  # Combine predictions
    ensemble_model = LinearRegression()  
    ensemble_model.fit(ensemble_X, y_test)

    # Make predictions using the ensemble model
    ensemble_predictions = ensemble_model.predict(ensemble_X)

    # Evaluate the ensemble model's performance
    ensemble_r_squared = r2_score(y_test, ensemble_predictions)
    ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
    return (ensemble_model, ensemble_r_squared, ensemble_mse)


def build_gradient_boosting_model(df_to_feed_into_models: pd.DataFrame):
    """
    Build and train a Gradient Boosting regression model with hyperparameter tuning.

    Parameters:
    - df_to_feed_into_models (pd.DataFrame): The input DataFrame containing predictors and the target variable.

    Returns:
    - Tuple[object, float, float]: A tuple containing the trained Gradient Boosting model, R-squared, and Mean Squared Error (MSE).
    """
    predictors = [col for col in df_to_feed_into_models.columns if col not in ['log_Papp_in_cm_sec']]
    X = df_to_feed_into_models[predictors]
    y = df_to_feed_into_models['log_Papp_in_cm_sec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=84)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    return (gradient_boosting_model, r_squared_gb, mse_gb)

def build_neural_net(df_to_feed_into_models: pd.DataFrame, l2_penalty=None):
    """
    Build and train a neural network model.

    Parameters:
    - df_to_feed_into_models (DataFrame): The input DataFrame containing relevant and other predictors, as well as the target variable.

    Returns:
    - Tuple[object, float, float]: A tuple containing the R-squared score and mean squared error of the trained neural network model.
    """

    # Here I'm making a subset of predictors that exclude all the chemical fingerprints 
    relevant_predictors = ['num_rotatable_bonds_Transformed',
                               'num_hydrogen_bond_donors_Transformed',
                               'num_hydrogen_bond_acceptors_Transformed',
                               'logp',
                               'num_rings_Transformed',
                               'num_aromatic_rings_Transformed',
                               'num_het_atoms_Transformed',
                               'MaxPartialCharge_Transformed',
                               'tpsa_Transformed',
                               'mol_weight_Transformed']

    # Here I'm storing all the predictors that are chemical fingerprints
    # other_predictors = [col for col in df_to_feed_into_models.columns if col not in relevant_predictors]
    other_predictors = [col for col in df_to_feed_into_models.columns if col not in relevant_predictors]
    other_predictors.remove('log_Papp_in_cm_sec')

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

    if l2_penalty != None:
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_combined.shape[1],),
                                    kernel_regularizer=l2(l2_penalty)),  # Add L2 regularization
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_penalty)),  # Add L2 regularization
                tf.keras.layers.Dense(1, kernel_regularizer=l2(l2_penalty))  # Output layer with L2 regularization
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
        return (model, r_squared_nn, mse_nn)

    else:
        # Define a list of l2_penalty values to test
        l2_penalty_values = [0.001, 0.01, 0.1, 1.0, 10.0]

        # Initialize variables to keep track of the best result
        best_l2_penalty = None
        best_r_squared = -np.inf  # Initialize with negative infinity
        best_mse = np.inf  # Initialize with positive infinity

        for l2_penalty in l2_penalty_values:
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_combined.shape[1],),
                                    kernel_regularizer=l2(l2_penalty)),  # Add L2 regularization
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(l2_penalty)),  # Add L2 regularization
                tf.keras.layers.Dense(1, kernel_regularizer=l2(l2_penalty))  # Output layer with L2 regularization
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

            if r_squared_nn > best_r_squared:
                best_r_squared = r_squared_nn
                best_mse = mse_nn
                best_l2_penalty = l2_penalty
            
            # Now rebuild the model with the best l2_penalty
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_combined.shape[1],),
                                    kernel_regularizer=l2(best_l2_penalty)),  # Add L2 regularization
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(best_l2_penalty)),  # Add L2 regularization
                tf.keras.layers.Dense(1, kernel_regularizer=l2(best_l2_penalty))  # Output layer with L2 regularization
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
            model.save("neural_net_permeability.h5")

            # Evaluate the model on the test data
            mse_nn = model.evaluate(X_test_combined_tf, y_test_tf)
            print(f"Mean Squared Error: {mse_nn}")

            y_pred_nn = model.predict(X_test_combined_tf)
            r_squared_nn = r2_score(y_test, y_pred_nn)
            print(f"R squared: {r_squared_nn}")
            
    return (model, r_squared_nn, mse_nn)


if __name__ == "__main__":
    # Build all the relevant models and store their accuracy metrics
    all_model_metrics_dict = {"Gradient Boosting Model": build_gradient_boosting_model(df_to_feed_into_models), "Ensemble: Gradient Boosting, Random Forest (interactions considered)":ensemble_gradient_plus_random_forest(df_to_feed_into_models, interactions=True),"Ensemble: Gradient Boosting, Random Forest (no interactions considered)":ensemble_gradient_plus_random_forest(df_to_feed_into_models), "Random Forest": build_random_forest_model(df_to_feed_into_models), "Neural Net": build_neural_net(df_to_feed_into_models), "LASSO regression": build_lasso_regression_model(df_to_feed_into_models) }
    all_model_metrics = pd.DataFrame.from_dict(all_model_metrics_dict, orient='index', columns=['Model','R-squared', 'MSE'])
    all_model_metrics.reset_index(inplace=True)
    all_model_metrics.rename(columns={'index': 'Model Type'}, inplace=True)
    all_model_metrics = all_model_metrics.sort_values(by='R-squared', ascending=False)
    print(all_model_metrics)
