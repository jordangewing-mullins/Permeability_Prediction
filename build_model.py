import tensorflow as tf
import matplotlib.pyplot as plt
from custom_functions import * 

file_path = 'perm.tsv'
df = pd.read_csv(file_path, delimiter='\t')


descriptor_columns = df['canonical_smiles'].apply(calculate_descriptors)
df = pd.concat([df, descriptor_columns], axis=1)

# apply some transformations to right skewed predictor data
df['tpsa_transformed'] = np.log1p(df['tpsa'])
df['mol_weight_transformed'] = np.log1p(df['mol_weight'])

df_new = df.copy()
df_new['log_Papp_in_cm_sec'] = np.log(df['Papp_in_cm_sec'])
columns_to_remove = ['Papp_in_cm_sec','mol_weight', 'tpsa','fingerprint', 'canonical_smiles','molregno']
df_new = df_new.drop(columns=columns_to_remove)

# Define your predictors and target variable
relevant_predictors = ['num_rotatable_bonds',
 'num_hydrogen_bond_donors',
 'num_hydrogen_bond_acceptors',
 'logp',
 'num_rings',
 'num_aromatic_rings',
 'num_het_atoms',
 'MaxPartialCharge']

other_predictors = [col for col in df_new.columns if col not in relevant_predictors]

X_relevant = df_new[relevant_predictors]
X_other = df_new[other_predictors]

# Split the data into training (70%) and test (30%) sets
X_train_relevant, X_test_relevant, X_train_other, X_test_other, y_train, y_test = train_test_split(
    X_relevant, X_other, df_new['log_Papp_in_cm_sec'], test_size=0.3, random_state=84
)

# Train a Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)  #  train the gradient boosting model, make predictions, and then use those predictions along with your original features as inputs to a neural network
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