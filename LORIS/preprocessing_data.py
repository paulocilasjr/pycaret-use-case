import os
import pandas as pd

# Set directory and create if not present
DIRECTORY = 'dataset_inputs'
os.makedirs(DIRECTORY, exist_ok=True)

def truncate_values(df):
    '''Truncate extreme values for TMB, Age, and NLR using .loc to avoid SettingWithCopyWarning'''
    df.loc[:, 'TMB'] = df['TMB'].clip(upper=50)
    df.loc[:, 'Age'] = df['Age'].clip(upper=85)
    df.loc[:, 'NLR'] = df['NLR'].clip(upper=25)
    print('Truncation completed')
    return df

def save_to_tsv(df, file_path):
    '''Save dataframe as TSV file'''
    df.to_csv(file_path, sep='\t', index=False)
    print(f'{file_path}: Save completed')

def read_sheet(file_path, sheet_name):
    '''Read an Excel sheet into a DataFrame'''
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
    print(f'{sheet_name}: Sheet read completed')
    return df

def select_and_truncate_columns(selected_columns, df):
    '''Select specific columns and truncate extreme values'''
    df_selected = df.loc[:, selected_columns].copy()
    df_selected = truncate_values(df_selected)
    print('Column selection completed')
    return df_selected

def get_feature_columns(response_present):
    '''Return selected columns based on response presence'''
    common_features = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] + \
                      [f'CancerType{i}' for i in range(1, 17)]
    
    if response_present == 'Response':
        return common_features + ['Response']
    
    return common_features

ALL_DATA_PATH = './dataset/AllData.xlsx'
sheets = ['Chowell_train', 'Chowell_test', 'MSK1']

for sheet in sheets:
    df = read_sheet(ALL_DATA_PATH, sheet)

    for feature_option in ['Response', 'No_Response']:
        selected_columns = get_feature_columns(feature_option)
        df_selected = select_and_truncate_columns(selected_columns, df)
        
        # Save processed data to file
        file_path = os.path.join(DIRECTORY, f'{sheet}_{feature_option}.tsv')
        save_to_tsv(df_selected, file_path)
