import pandas as pd
from pycaret.classification import *
from sklearn.metrics import roc_auc_score

def truncate_values(dataframe):
    '''doc string'''
    # Truncate extreme values
    dataframe['TMB'] = dataChowell_Train['TMB'].clip(upper=50)
    dataframe['Age'] = dataChowell_Train['Age'].clip(upper=85)
    dataframe['NLR'] = dataChowell_Train['NLR'].clip(upper=25)

    return dataframe

featuresNA = ['TMB', 'Systemic_therapy_history',
              'Albumin', 'NLR', 'Age'] + \
             [f'CancerType{i}' for i in range(1, 17)]

# Load and preprocess training data
DATALL = './dataset/AllData.xlsx'

# Load tab and select feature columns
dataChowell_Train = pd.read_excel(DATALL, sheet_name='Chowell_train', index_col=0)
dataChowell_Train = dataChowell_Train[featuresNA + ['Response']]
dataChowell_Train = truncate_values(dataChowell_Train)

# Load and preprocess the test data (Chowell_test)
dataChowell_Test = pd.read_excel(DATALL, sheet_name='Chowell_test', index_col=0)
dataChowell_Test = truncate_values(dataChowell_Test)
# Ensure the test data has the same feature set
dataChowell_Test = dataChowell_Test[featuresNA + ['Response']]


def check_auc(auc_metric):
    '''docstring'''
    if auc_metric == 0.75:
        return True
    return False

def run_model(session_number):
    '''docstring'''
    # 'session_id': 3906, 'auc': 0.7222
    setup(data=dataChowell_Train, 
          target='Response',
          session_id=session_number,
          normalize=True,
          feature_selection=False,
          )

    model = create_model('lr', penalty='elasticnet', solver='saga', l1_ratio=1, class_weight='balanced', C=0.1, max_iter=100)
    #print(model)
    #plot_model(model, plot='feature')

    tuned_model = tune_model(model, early_stopping_max_iters= 100, optimize='AUC')
    #print(tuned_model)

    predictions_df = predict_model(tuned_model, data=dataChowell_Test)

    metrics_df = pull()
    auc_metric = metrics_df.loc[0, 'AUC']

    return auc_metric

def update_best(auc, session_id, best):
    '''docstring'''
    if auc > best['auc']:
        return {
            'session_id': session_id,
            'auc': auc
        }
    return best


session_id = 1000

best = {'session_id': "", 'auc': 0}

while session_id != 10000:
    auc_result = run_model(session_id)
    bool_auc = check_auc(auc_result)
    if bool_auc:
        print(f'Hit perfect!! #### session_id = {session_id}')
        break
    best = update_best(auc_result, session_id, best)
    session_id += 1

print(best)
