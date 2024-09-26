import os
import time
import pandas as pd
from pycaret.classification import *
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold


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

# Configure RepeatedStratifiedKFold for 5 folds and 2000 repeats
cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=2000, random_state=42)

# Train the final model using the entire training dataset
print("Training on the full dataset...")
setup(data=dataChowell_Train, 
      target=phenoNA, 
      session_id=randomSeed, 
      normalize=True, 
      feature_selection=False, 
      fold_strategy=cv_strategy)

best_model = create_model('lr', penalty='elasticnet', solver='saga', l1_ratio=1, class_weight='balanced', C=0.1, max_iter=100)
print(best_model)
plot_model(best_model, plot='feature')

# Load and preprocess the test data (Chowell_test)
dataChowell_Test = pd.read_excel(dataALL_fn, sheet_name='Chowell_test', index_col=0)
dataChowell_Test = TruncateValues(dataChowell_Test)

# Ensure the test data has the same feature set
dataChowell_Test = dataChowell_Test[featuresNA + [phenoNA]]

# Predict on the external test data using the final tuned model
predictions_df = predict_model(best_model, data=dataChowell_Test)

print(f'All done! Time used: {time.time() - start_time}')
