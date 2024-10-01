import os
import pandas as pd

directory = 'dataset_inputs'
if not os.path.exists(directory):
    os.makedirs(directory)

def truncate_values(dataframe):
    '''truncate extreme values for tmb, age, and nlr'''
    dataframe.loc[:, 'TMB'] = dataframe['TMB'].clip(upper=50)
    dataframe.loc[:, 'Age'] = dataframe['Age'].clip(upper=85)
    dataframe.loc[:, 'NLR'] = dataframe['NLR'].clip(upper=25)
    print('truncate completed')
    return dataframe

def save_in_file(tsv_file, tsv_file_name):
    '''save tsv file dataset'''
    tsv_file.to_csv(tsv_file_name, sep='\t', index=False)
    print(f'{tsv_file_name}: save completed')

def read_sheet(data_file, sheet_name):
    dataframe = pd.read_excel(data_file, sheet_name=sheet_name, index_col=0)
    print('read sheet completed')
    return dataframe 

def column_selections(selected_columns, dataframe):
    dataframe = dataframe.loc[:, selected_columns]
    dataframe = truncate_values(dataframe)
    print('column selection completed')
    return dataframe

features = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] + \
          [f'CancerType{i}' for i in range(1, 17)]

features_response = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] + \
                   [f'CancerType{i}' for i in range(1, 17)] + ['Response']

# Load and preprocess training data
data_all = './dataset/AllData.xlsx'

list_of_sheets = ['Chowell_train', 'Chowell_test', 'MSK1']

for sheet in list_of_sheets:
    dataframe = read_sheet(data_all, sheet)
    
    dataframe_no_response = column_selections(features, dataframe)
    file_name = f'{directory}/{sheet}_without_response.tsv'
    save_in_file(dataframe_no_response, file_name)
    
    dataframe_response = column_selections(features_response, dataframe)
    file_name = f'{directory}/{sheet}_with_response.tsv'
    save_in_file(dataframe_response, file_name)
