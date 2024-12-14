import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 heat_map.py <info_models_path_csv>")
        sys.exit(1)
            
    file_path = sys.argv[1]
        
    df = pd.read_csv(file_path)
    
    columns_to_average = ["size", "r2", "mse", "r2-cpu", "r2-mem", "r2-thrpt", "mse-cpu", "mse-mem", "mse-thrpt", "training-time"]
    models = ["GRU", "LR", "MLP", "RF", "XGBoost"]

    # Iterar sobre los modelos y pasos de tiempo
    for model in models:
        for step in range(1, 30):
            for colum in columns_to_average:
                # Filtrar los datos para el modelo y el paso de tiempo
                filtered_data = df[(df["name"] == model) & (df["time-step"] == step)]
                
                # Calcular el promedio de la columna de interés
                if not filtered_data.empty:  # Asegurarse de que no esté vacío
                    average = filtered_data[colum].mean()
                    # Mostrar resultados
                    print(f"Model {model}, Steptime: {step}, Metric: {colum} -> Average: {average:.4f}")
                else:
                    print(f"Model {model}, Steptime: {step}, Metric: {colum} -> No data available")

if __name__ == "__main__":
    main()
