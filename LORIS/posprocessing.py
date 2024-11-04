import requests
from bs4 import BeautifulSoup
import re

# Evaluation LORIS LLR6 Model Numbers from Chang et al., 2024
LLR6 = {
    'chowell_tran': {
        'auc': [0.7441, 0.7441 - 1.96 * (0.008886 / (10000 ** 0.5)), 0.7441 + 1.96 * (0.008886 / (10000 ** 0.5))],
        'prauc': [0.5501, 0.0159, 0.4868, 0.6210],
        'accuracy': [0.6928, 0.0088, 0.6563, 0.7263],
        'f1': 0.5495,
        'mcc': 0.4812,
        'BA': 0.6407,
        'performance': 0.6283,
    },
    'chowell_test': {
        'auc': [0.7155, 0.7155 - 1.96 * (0.035784 / (10000 ** 0.5)), 0.7155 + 1.96 * (0.035784 / (10000 ** 0.5))],
        'prauc': [0.5302, 0.0599, 0.2869, 0.7404],
        'accuracy': [0.6803, 0.0305, 0.5648, 0.7927],
        'f1': 0.5289,
        'mcc': 0.4652,
        'BA': 0.6176,
        'performance': 0.6072,
    },
    'msk1': {
        'auc': [0.70, 0.65, 0.75]
    },
}

def extract_auc_value(script_tags):
    for script in script_tags:
        if script.string and 'AUC =' in script.string:
            match = re.search(r'AUC\s*=\s*(\d+\.\d+)', script.string)
            if match:
                return float(match.group(1))
    return None


def compare_metric(value, std, ci_lower=None, ci_upper=None, metric_name=""):
    if value > std:
        comparison = "higher"
    elif value < std:
        comparison = "lower"
    else:
        comparison = "equal"

    print(f"The extracted {metric_name} value {value:.4f} is {comparison} than the standard {metric_name} value {std:.4f}.")

    # Only check the confidence interval if both ci_lower and ci_upper are provided
    if ci_lower is not None and ci_upper is not None:
        if ci_lower <= value <= ci_upper:
            print(f"The extracted {metric_name} value {value:.4f} is within the confidence interval ({ci_lower}, {ci_upper}).")
        else:
            print(f"The extracted {metric_name} value {value:.4f} is outside the confidence interval ({ci_lower}, {ci_upper}).")

def AUCMetric(soup, dataset_name):
    auc_value = extract_auc_value(soup.find_all('script'))
    if auc_value is None:
        print("AUC value not found.")
        return
    
    try:
        auc_data = LLR6[dataset_name]['auc']
        compare_metric(auc_value, auc_data[0], auc_data[1], auc_data[2], "AUC")
    except KeyError:
        print(f"No AUC information available for {dataset_name}.")

def extract_metrics(script_content, dataset_name):
    pattern = r'"name":\s*"(Precision|Recall|F-score)",\s*"type":\s*"scatter",\s*"x":\s*\[[0-9, ]+\],\s*"y":\s*\[([0-9.]+),\s*([0-9.]+)\]'
    matches = re.findall(pattern, script_content)
    
    results = {name: (round(float(y1), 4), round(float(y2), 4)) for name, y1, y2 in matches}
    print("Extracted Values:", results)
    
    f_score = results.get('F-score', (None,))[0]
    f1_score = LLR6.get(dataset_name, {}).get('f1')
    if f_score is not None and f1_score is not None:
        compare_metric(f_score, f1_score, metric_name="F-score")

def OtherMetrics(soup, dataset_name):
    script = soup.find('script', string=re.compile('Plotly.newPlot'))
    if script and script.string:
        extract_metrics(script.string, dataset_name)
    else:
        print('Plotly.newPlot script not found.')

def GetHTML(url, plot_type, dataset_name):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch content: {e}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    if plot_type == 'auc':
        AUCMetric(soup, dataset_name)
    elif plot_type == 'metrics':
        OtherMetrics(soup, dataset_name)
    else:
        print("Unknown plot type.")

# URLs for AUC and other metrics
dataset_name = 'chowell_test'
auc_plot_url = 'https://usegalaxy.org/api/datasets/f9cad7b01a472135fa4258a31fe1755d/display?to_ext=html'
metrics_url = 'https://usegalaxy.org/api/datasets/f9cad7b01a472135f5481594fb28c9e4/display?to_ext=html'

# Call functions to fetch HTML and extract metrics
GetHTML(auc_plot_url, 'auc', dataset_name)
GetHTML(metrics_url, 'metrics', dataset_name)
