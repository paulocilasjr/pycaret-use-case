import os
import time
import pandas as pd
from pycaret.classification import setup, create_model, tune_model, plot_model, predict_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import RepeatedKFold

# Check if the folder exists
folder_name = 'pycaret_outputs'
if not os.path.exists(folder_name):
    # Create the folder if it doesn't exist
    os.makedirs(folder_name)


def TruncateValues(dataframe):
    # Truncate extreme values
    dataframe['TMB'] = dataChowell_Train['TMB'].clip(upper=50)
    dataframe['Age'] = dataChowell_Train['Age'].clip(upper=85)
    dataframe['NLR'] = dataChowell_Train['NLR'].clip(upper=25)

    return dataframe


start_time = time.time()

randomSeed = 1
train_size = 0.8
phenoNA = 'Response'
LLRmodelNA = 'LLR6'

featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] + \
             [f'CancerType{i}' for i in range(1, 17)]

# Load and preprocess training data
dataALL_fn = './dataset/AllData.xlsx'

# Load tab and select feature columns
dataChowell_Train = pd.read_excel(dataALL_fn, sheet_name='Chowell_train', index_col=0)
dataChowell_Train = dataChowell_Train[featuresNA + [phenoNA]]
dataChowell_Train = TruncateValues(dataChowell_Train)

print(f'Chowell patient number (training): {dataChowell_Train.shape[0]}')

# Train the final model using the entire training dataset
print("Training on the full dataset...")

# Original
#setup(data=dataChowell_Train, target=phenoNA, session_id=randomSeed, normalize=True, feature_selection=False, use_gpu=True)

cv = RepeatedKFold(n_splits=5, n_repeats=2000, random_state=42)

# Changing to get similar to the paper
setup(data=dataChowell_Train, target=phenoNA, session_id=randomSeed,
      normalize=True, normalize_method='zscore',
      fold_strategy=cv, use_gpu=True)

LLR6_model = create_model('lr', penalty='elasticnet',
                          solver='saga', l1_ratio=1,
                          class_weight='balanced',
                          C=0.1, max_iter=100)

print(f"model created: {LLR6_model}")


# Define the custom grid for logistic regression (solver='saga', penalty='elasticnet', etc.)
param_grid = {
    'solver': ['saga'],
    'penalty': ['elasticnet'],
    'class_weight': ['balanced'],
    'l1_ratio': [i/10.0 for i in range(0, 11)],
    'max_iter': [i for i in range(100, 1100, 100)],
    'C': [10**i for i in range(-3, 4)]}

# Tune the final model on the entire dataset
tuned_LLR6_model = tune_model(LLR6_model, optimize='AUC', custom_grid=param_grid)
plot_model(tuned_LLR6_model, plot='feature')
print(f"Done tune: {tuned_LLR6_model}")

# Save the final model trained on the entire dataset
# save_model(tuned_LLR6_model, f'./pycaret_outputs/LLR_full_pancancer_model')
print("#################################################")
# Load and preprocess the test data (Chowell_test)
dataChowell_Test = pd.read_excel(dataALL_fn, sheet_name='Chowell_test', index_col=0)
dataChowell_Test = TruncateValues(dataChowell_Test)

# Ensure the test data has the same feature set
dataChowell_Test = dataChowell_Test[featuresNA + [phenoNA]]

print(f'Number of rows: {len(dataChowell_Test)}')

# Predict on the external test data using the final tuned model
predictions_df = predict_model(tuned_LLR6_model, data=dataChowell_Test)

# Evaluation of the prediction
print('Starting evaluation of the prediction')
# Dashboard function
# dashboard(tuned_LLR6_model, display_format='inline')

# Save the predictions
#predictions_df.to_csv(f'./pycaret_outputs/LLR6_pan_predictions.csv', index=False)
# print(f'Predictions saved')

print(f'All done! Time used: {time.time() - start_time}')
