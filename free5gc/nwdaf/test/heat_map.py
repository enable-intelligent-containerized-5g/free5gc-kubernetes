import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_name = "models_info_dataset_NF_LOAD_AMF_60s_1733807760_1733915795_total-steps-30"
data_path = f"models-info/{dataset_name}.csv"
df = pd.read_csv(data_path)

df['mse-cpu'] = np.sqrt(df['mse-cpu'])
df['mse-mem'] = np.sqrt(df['mse-mem'])
df['mse'] = np.sqrt(df['mse'])

# Create the structure
# ascending -> better Top, descending -> better Bottom, 
metrics_structure = [
    # {'column': 'size', 'title': 'Size', 'trend': 'descending', 'decimals': 0},
    {'column': 'r2', 'title': 'R2', 'trend': 'ascending', 'decimals': 3},
    {'column': 'mse', 'title': 'MSE', 'trend': 'descending', 'decimals': 2},
    {'column': 'r2-cpu', 'title': 'R2 CPU', 'trend': 'ascending', 'decimals': 3},
    {'column': 'r2-mem', 'title': 'R2 Memory', 'trend': 'ascending', 'decimals': 3},
    {'column': 'r2-thrpt', 'title': 'R2 Throughput', 'trend': 'ascending', 'decimals': 2},
    {'column': 'mse-cpu', 'title': 'MSE CPU', 'trend': 'descending', 'decimals': 8},
    {'column': 'mse-mem', 'title': 'MSE Memory', 'trend': 'descending', 'decimals': 8},
    {'column': 'mse-thrpt', 'title': 'MSE Throughput', 'trend': 'descending', 'decimals': 2},
    {'column': 'training-time', 'title': 'Training time', 'trend': 'descending', 'decimals': 4}
]


for metric in metrics_structure:
    metric_column = metric['column']
    metric_title = metric['title']
    metric_decimals = metric['decimals']
    
    color = 'RdYlGn'
    if metric['trend'] == 'descending':
        color = 'RdYlGn_r'
        
    plt.close('all')
    # Create data table
    pivot_table = df.pivot_table(values=metric_column, index='time-step', columns='name')
    pivot_table = pivot_table.iloc[::-1]
    # Create HeatMap
    plt.figure(figsize=(12, 8))
    heat_map = sns.heatmap(pivot_table, annot=True, fmt=f'.{metric_decimals}f', cmap=color, linewidths=0.5)
    plt.yticks(rotation=0)
    plt.title(f'{metric_title} Heatmap')
    plt.xlabel('Model')
    plt.ylabel('Time sequence length')
    # Show the plot
    # plt.show()
    plt.savefig(f"figures-heat-map/figure_{dataset_name}_metric-{metric_column}.png", bbox_inches='tight', pad_inches=0.01)
    
