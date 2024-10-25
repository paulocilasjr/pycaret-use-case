import os
import pandas as pd

# Set directory and create if not present
DIRECTORY = 'dataset_inputs'
os.makedirs(DIRECTORY, exist_ok=True)

def truncate_values(df):
    """Truncate extreme values for TMB, Age, and NLR to avoid SettingWithCopyWarning."""
    df['TMB'] = df['TMB'].clip(upper=50)
    df['Age'] = df['Age'].clip(upper=85)
    df['NLR'] = df['NLR'].clip(upper=25)
    return df

def save_to_tsv(df, file_path):
    """Save DataFrame as TSV file."""
    try:
        df.to_csv(file_path, sep='\t', index=False)
    except Exception as e:
        print(f'Error saving {file_path}: {e}')

def read_sheet(file_path, sheet_name):
    """Read an Excel sheet into a DataFrame with error handling."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)
        return df
    except FileNotFoundError:
        print(f'Error: The file {file_path} does not exist.')
    except ValueError:
        print(f'Error: Sheet {sheet_name} not found in {file_path}.')

def select_and_truncate_columns(selected_columns, df):
    """Select specified columns and truncate extreme values."""
    df_selected = df.loc[:, selected_columns].copy()
    df_selected = truncate_values(df_selected)
    return df_selected

def get_feature_columns(response_present):
    """Return selected columns based on response presence."""
    common_features = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] + \
                      [f'CancerType{i}' for i in range(1, 17)]
    return common_features + ['Response'] if response_present == 'Response' else common_features

ALL_DATA_PATH = './dataset/AllData.xlsx'
sheets = ['Chowell_train', 'Chowell_test', 'MSK1']

for sheet in sheets:
    df = read_sheet(ALL_DATA_PATH, sheet)
    
    if df is not None:  # Check if DataFrame was read successfully
        for feature_option in ['Response', 'No_Response']:
            selected_columns = get_feature_columns(feature_option)
            df_selected = select_and_truncate_columns(selected_columns, df)
            
            # Save processed data to file
            file_path = os.path.join(DIRECTORY, f'{sheet}_{feature_option}.tsv')
            save_to_tsv(df_selected, file_path)
