import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_averages(df, metric_structure, info_file_name):
    # Seleccionar las columnas de métricas
    columns_to_plot = [metric['column'] for metric in metric_structure]

    # Calcular los promedios agrupados por modelo
    averages = df.groupby("name")[columns_to_plot].mean()

    # Crear un gráfico por cada métrica
    for metric in metric_structure:
        # Crear figura y ejes
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6))

        # Normalización para el colormap
        norm = Normalize(vmin=averages[metric['column']].min(), vmax=averages[metric['column']].max())
        cmap = plt.cm.RdYlGn if metric['trend'] != 'descending' else plt.cm.RdYlGn_r

        # Graficar barras
        colors = [cmap(norm(value)) for value in averages[metric['column']]]
        bars = ax.bar(
            averages.index, 
            averages[metric['column']], 
            color=colors, 
            edgecolor="black"
        )

        # Añadir etiquetas a las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                height, 
                f"{height:.{metric['decimals']}f}", 
                ha='center', va='bottom', fontsize=12, color='black'
            )

        # Añadir título y etiquetas
        ax.set_title(f"Average {metric['title']} by model", fontsize=14)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylabel(metric['unit'], fontsize=12)

        # Añadir barra de color
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Asociar datos al ScalarMappable
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(metric['unit'], fontsize=12)

        # Ajustar el diseño y guardar la figura
        plt.tight_layout()
        plt.savefig(f"figures-average/figure_{info_file_name}_metric-{metric['column']}.pdf", 
                    bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)  # Cerrar la figura para liberar memoria

def custom_fmt(val, decimals):
    if val == 0:
        return f'{0:.0f}'
    else:
        return f'{val:.{decimals}f}'

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 heat_map.py <directory-path> <info_models_path_csv>")
        sys.exit(1)
            
    directory_path = sys.argv[1]
    info_file_name = sys.argv[2]
        
    # dataset_name = "models_info_dataset_NF_LOAD_AMF_60s_1734070080_1734091260_total-steps-30"
    data_path = f"{directory_path}{info_file_name}.csv"
    df = pd.read_csv(data_path)
        
    very_small_value = 0
    columns_to_modify = ['r2', 'r2-cpu', 'r2-mem', 'r2-thrpt']
    mask = df[columns_to_modify] < 0
    df.loc[:, columns_to_modify] = df.loc[:, columns_to_modify].mask(mask, very_small_value)

    # Create the structure
    # ascending -> better Top, descending -> better Bottom, 
    metrics_structure = [
        {'column': 'size', 'title': 'Size', 'trend': 'descending', 'decimals': 0, 'unit': 'Bytes'},
        {'column': 'r2', 'title': 'R2', 'trend': 'ascending', 'decimals': 3, 'unit': 'R2'},
        {'column': 'mse', 'title': 'RMSE', 'trend': 'descending', 'decimals': 2, 'unit': 'RMSE'},
        {'column': 'r2-cpu', 'title': 'R2 CPU', 'trend': 'ascending', 'decimals': 3, 'unit': 'R2 CPU'},
        {'column': 'r2-mem', 'title': 'R2 Memory', 'trend': 'ascending', 'decimals': 3, 'unit': 'R2 Memory'},
        {'column': 'r2-thrpt', 'title': 'R2 Throughput', 'trend': 'ascending', 'decimals': 3, 'unit': 'R2 Throughput'},
        {'column': 'mse-cpu', 'title': 'RMSE CPU', 'trend': 'descending', 'decimals': 7, 'unit': 'CPU usage (%)'},
        {'column': 'mse-mem', 'title': 'RMSE Memory', 'trend': 'descending', 'decimals': 7, 'unit': 'Memory usage (%)'},
        {'column': 'mse-thrpt', 'title': 'RMSE Throughput', 'trend': 'descending', 'decimals': 2, 'unit': 'Bytes/s'},
        {'column': 'training-time', 'title': 'Training time', 'trend': 'descending', 'decimals': 4, 'unit': 'Seconds'}
    ]

    plot_averages(df, metrics_structure, info_file_name)
    
    plt.close('all')
    df['rolling_mse'] = df.groupby('name')['mse'].transform(lambda x: x.rolling(window=2, min_periods=1).mean())

    # Graficando
    sns.lineplot(data=df, x="time-step", y="rolling_mse", hue="name", marker="o")

    plt.title("Tendencia Suavizada (Promedio Móvil) por Modelo", fontsize=16)
    plt.xlabel("Time-Step", fontsize=14)
    plt.ylabel("MSE (Promedio Móvil)", fontsize=14)
    plt.legend(title="Modelo")
    plt.show()
    plt.show()
    

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
        heat_map = sns.heatmap(pivot_table, 
                               annot=True, 
                               fmt=f'.{metric_decimals}f', 
                               cmap=color, 
                               linewidths=0.5)
        # Custom format
        for text in heat_map.texts:
            val = float(text.get_text())  # Obtener el valor del texto
            text.set_text(custom_fmt(val, metric_decimals)) 
        plt.yticks(rotation=0)
        plt.title(f'{metric_title} Heatmap')
        plt.xlabel('Model')
        plt.ylabel('Time window length')
        # Save the plot
        # plt.show()
        plt.savefig(f"figures-heat-map/figure_{info_file_name}_metric-{metric_column}.pdf", bbox_inches='tight', pad_inches=0.05)
       
if __name__ == "__main__":
    main()
