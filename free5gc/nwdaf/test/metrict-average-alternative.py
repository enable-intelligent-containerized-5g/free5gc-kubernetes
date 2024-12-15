import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def color_scheme(value, metric, min_value, max_value, p33, p66):
    """
    Devuelve el color de la barra según el valor y la métrica.
    """
    if metric in ["mse", "mse-cpu", "mse-mem", "mse-thrpt", "training-time"]:
        # Para estas métricas, valores bajos son mejores
        if value <= p33:
            return '#66bb6a'  # Verde
        elif value <= p66:
            return '#fbc02d'  # Amarillo
        else:
            return '#ef5350'  # Rojo
    elif metric in ["r2", "r2-cpu", "r2-mem", "r2-thrpt"]:
        # Para r2, valores altos son mejores
        if value >= p66:
            return '#66bb6a'  # Verde
        elif value >= p33:
            return '#fbc02d'  # Amarillo
        else:
            return '#ef5350'  # Rojo
    else:
        # Otros colores por defecto
        return '#90a4ae'

def plot_metric_windows(df, stats, model_averages, mse_columns, r2_columns, other_columns):
    """
    Genera ventanas con gráficos de barras para cada métrica agrupada.
    """
    metrics = [
        ("MSE Metrics", mse_columns),
        ("R2 Metrics", r2_columns),
        ("Other Metrics", other_columns)
    ]

    for metric_name, columns in metrics:
        # Crear una figura para cada ventana de métricas
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))  # 4 subgráficas por ventana

        for idx, col in enumerate(columns):
            avg_values = model_averages[col].values
            x_positions = np.arange(len(avg_values))
            bars = axes[idx].bar(x_positions, avg_values, color='gray')  # Colores iniciales

            # Ajustar colores de las barras y mostrar los valores promedio
            for bar, avg in zip(bars, avg_values):
                min_value = stats[col]['min']
                max_value = stats[col]['max']
                p33 = stats[col]['p33']
                p66 = stats[col]['p66']
                color = color_scheme(avg, col, min_value, max_value, p33, p66)
                bar.set_color(color)

                # Ajustar posición del texto según la altura de la barra
                yval = bar.get_height()
                if yval > max(avg_values) * 0.2:
                    # Si la barra es alta, coloca el texto dentro
                    axes[idx].text(bar.get_x() + bar.get_width() / 2, yval - (yval * 0.1), f'{yval:.4f}', 
                                   ha='center', va='top', fontsize=10, fontweight='bold', color='white')
                else:
                    # Si la barra es baja, coloca el texto justo encima
                    axes[idx].text(bar.get_x() + bar.get_width() / 2, yval + (yval * 0.05), f'{yval:.4f}', 
                                   ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

            # Personalización de cada gráfico
            axes[idx].set_title(f'{col}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Modelo', fontsize=12)
            axes[idx].set_ylabel('Promedio', fontsize=12)
            axes[idx].set_xticks(x_positions)
            axes[idx].set_xticklabels(model_averages.index, rotation=45, ha='right', fontsize=10)
            axes[idx].grid(axis='y', linestyle='--', alpha=0.7)

            # Ajuste dinámico del límite superior del eje y
            axes[idx].set_ylim(0, max(avg_values) * 1.2)  # Agregar un 20% de margen superior

        # Ajustar el diseño de la ventana
        plt.tight_layout()

        # Mostrar la ventana de métricas
        plt.suptitle(metric_name, fontsize=16, fontweight='bold', y=1.02)
        plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 heat_map.py <info_models_path_csv>")
        sys.exit(1)
            
    file_path = sys.argv[1]
        
    # Leer el CSV de entrada
    df = pd.read_csv(file_path)
    
    # Definir las columnas y modelos a analizar
    mse_columns = ["mse", "mse-cpu", "mse-mem", "mse-thrpt"]
    r2_columns = ["r2", "r2-cpu", "r2-mem", "r2-thrpt"]
    other_columns = ["training-time"]

    models = ["GRU", "LR", "MLP", "RF", "XGBoost"]

    # Calcular estadísticas generales
    stats = {}
    for col in mse_columns + r2_columns + other_columns:
        stats[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'p33': np.percentile(df[col], 33),
            'p66': np.percentile(df[col], 66)
        }
    
    # Calcular promedios por modelo y métrica
    model_averages = df.groupby(['name'])[mse_columns + r2_columns + other_columns].mean()

    # Graficar métricas
    plot_metric_windows(df, stats, model_averages, mse_columns, r2_columns, other_columns)

if __name__ == "__main__":
    main()
