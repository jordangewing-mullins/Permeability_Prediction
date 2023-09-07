# Cheminformatics_Takehome
Making a predictive model to evaluate the permeability of small molecules.

# Create the virtual environment
> conda env create -f environment.yml

# Install dependencies
> pip install -r requirements.txt

# Run exploratory data analysis script
I prefer to use iPython in VS Code, so I use the command:
> run Exploratory_Data_Analysis.py

This step should take ~2 minutes, since the featurization happens here. The script will generate three plots stored as .png files and will create a CSV file of cleaned up and featurized data to pass onto build_model.py.

# Run the script to build the model
Also from iPython you can use the command:
> run build_model.py

This will build a few different models and will then print out the dataframe containing the accuracy metrics for each model tested. This step should take ~10 minutes. I tried to train my model on a GPU, but the libraries I used appeared to be incompatible with GPU processing.

# Interact with the command line to get predictions for your SMILES string
> run permeability_prediction.py "<your_SMILES_string_here>"
