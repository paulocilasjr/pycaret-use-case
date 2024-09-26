import time
import pandas as pd
from pycaret.classification import setup, create_model, tune_model, predict_model, plot_model
from sklearn.model_selection import RepeatedStratifiedKFold

def truncate_values(dataframe):
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

featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] \
            + [f'CancerType{i}' for i in range(1, 17)]

# Load and preprocess training data
dataALL_fn = './dataset/AllData.xlsx'

# Load tab and select feature columns
dataChowell_Train = pd.read_excel(dataALL_fn,
                                  sheet_name='Chowell_train',
                                  index_col=0)

dataChowell_Train = dataChowell_Train[featuresNA + [phenoNA]]

dataChowell_Train = truncate_values(dataChowell_Train)

print(f'Chowell patient number (training): {dataChowell_Train.shape[0]}')

# Configure RepeatedStratifiedKFold for 5 folds and 2000 repeats
cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=2000, random_state=42)

# Train the final model using the entire training dataset
print("Training on the full dataset...")
setup(data=dataChowell_Train,
      target=phenoNA,
      session_id=randomSeed,
      fold_strategy=cv_strategy)

model = create_model('lr')
print(model)
print("model created")

plot_model(model, plot='feature')

# Load and preprocess the test data (Chowell_test)
dataChowell_Test = pd.read_excel(dataALL_fn,
                                 sheet_name='Chowell_test',
                                 index_col=0)
dataChowell_Test = truncate_values(dataChowell_Test)

# Ensure the test data has the same feature set
dataChowell_Test = dataChowell_Test[featuresNA + [phenoNA]]

# Predict on the external test data using the final tuned model
predictions_df = predict_model(model, data=dataChowell_Test)

# Evaluation of the prediction
print('Starting evaluation of the prediction')
# Dashboard function

print(f'All done! Time used: {time.time() - start_time}')
