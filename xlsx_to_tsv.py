import pandas as pd

def truncate_values(dataframe):
    '''truncate extreme values for tmb, age, and nlr'''
    dataframe['TMB'] = dataframe['TMB'].clip(upper=50)
    dataframe['Age'] = dataframe['Age'].clip(upper=85)
    dataframe['NLR'] = dataframe['NLR'].clip(upper=25)
    return dataframe

featuresna = ['TMB', 'Systemic_therapy_history',
              'Albumin', 'NLR', 'Age'] + \
             [f'CancerType{i}' for i in range(1, 17)]

# load and preprocess training data
datall = 'AllData.xlsx'

# load and preprocess the test data (chowell_train)
datachowell_train = pd.read_excel(datall, sheet_name='Shim_NSCLC', index_col=0)
datachowell_train = datachowell_train[featuresna + ['Response']]
#datachowell_train = datachowell_train[featuresna]


datachowell_train = truncate_values(datachowell_train)

# Save the selected data as a TSV file
tsv_file = 'LORIS_shim_file_w_Response.tsv'
datachowell_train.to_csv(tsv_file, sep='\t', index=False)

