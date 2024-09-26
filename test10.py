import pandas as pd
from pycaret.classification import *

def TruncateValues(dataframe):
    # Truncate extreme values
    dataframe['TMB'] = dataChowell_Train['TMB'].clip(upper=50)
    dataframe['Age'] = dataChowell_Train['Age'].clip(upper=85)
    dataframe['NLR'] = dataChowell_Train['NLR'].clip(upper=25)

    return dataframe

featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] + [f'CancerType{i}' for i in range(1, 17)]

# Load and preprocess training data
dataALL_fn = './dataset/AllData.xlsx'

# Load tab and select feature columns
dataChowell_Train = pd.read_excel(dataALL_fn, sheet_name='Chowell_train', index_col=0)
dataChowell_Train = dataChowell_Train[featuresNA + ['Response']]
dataChowell_Train = TruncateValues(dataChowell_Train)

print(dataChowell_Train)

setup(data=dataChowell_Train, 
      target='Response', 
      session_id=123, 
      normalize=True, 
      feature_selection=False, 
      )
model = create_model('lr', penalty='elasticnet', solver='saga', l1_ratio=1, class_weight='balanced', C=0.1, max_iter=100)
print(model)
plot_model(model, plot='feature')

# Load and preprocess the test data (Chowell_test)
dataChowell_Test = pd.read_excel(dataALL_fn, sheet_name='Chowell_test', index_col=0)
dataChowell_Test = TruncateValues(dataChowell_Test)

# Ensure the test data has the same feature set
dataChowell_Test = dataChowell_Test[featuresNA + ['Response']]

predictions_df = predict_model(model, data=dataChowell_Test)
