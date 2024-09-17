import sys
import time
import pandas as pd
import numpy as np
from pycaret.classification import *

def TruncateValues (dataframe):
    # Truncate extreme values
    dataframe['TMB'] = dataChowell_Train['TMB'].clip(upper=50)
    dataframe['Age'] = dataChowell_Train['Age'].clip(upper=85)
    dataframe['NLR'] = dataChowell_Train['NLR'].clip(upper=25)

start_time = time.time()

randomSeed = 1
train_size = 0.8
phenoNA = 'Response'
LLRmodelNA = 'LLR6' 

featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] + [f'CancerType{i}' for i in range(1, 17)]

# Load and preprocess training data
dataALL_fn = './dataset/AllData.xlsx'

# Load tab and select feature columns
dataChowell_Train = pd.read_excel(dataALL_fn, sheet_name='Chowell_train', index_col=0)
dataChowell_Train = dataChowell_Train[featuresNA + [phenoNA]]
dataChowell_Train = TruncateValues(dataChowell_Train)

print(f'Chowell patient number (training): {dataChowell_Train.shape[0]}')

# Train the final model using the entire training dataset
print("Training on the full dataset...")
setup(data=dataChowell_Train, target=phenoNA, session_id=randomSeed, normalize=True, feature_selection=False)

LLR6_model = create_model('lr', penalty='elasticnet', solver='saga', l1_ratio=1, class_weight='balanced', C=0.1)
print("model created")

# Tune the final model on the entire dataset
tuned_LLR6_model = tune_model(LLR6_model, n_iter=1000, optimize='AUC')
print("Done tune")

# Save the final model trained on the entire dataset
save_model(tuned_LLR6_model, f'./pycaret_outputs/LLR_full_pancancer_model')

# Load and preprocess the test data (Chowell_test)
dataChowell_Test = pd.read_excel(dataALL_fn, sheet_name='Chowell_test', index_col=0)
dataChowell_Test = TruncateValues(dataChowell_Test)

# Ensure the test data has the same feature set
dataChowell_Test = dataChowell_Test[featuresNA + [phenoNA]]

# Predict on the external test data using the final tuned model
predictions_df = predict_model(tuned_LLR6_model, data=dataChowell_Test)

# Evaluation of the prediction 
print ('Starting evaluation of the prediction')
# Dashboard function
dashboard(tuned_LLR6_model, display_format='inline')

# Save the predictions
predictions_df.to_csv(f'./pycaret_outputs/LLR6_pan_predictions.csv', index=False)
print(f'Predictions saved')

print(f'All done! Time used: {time.time() - start_time}')