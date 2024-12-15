import os
import math
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam, SGD

# Configuración básica del logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula las métricas de rendimiento (RMSE y R²) para los datos predichos y reales.
    """
    return {
        'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }


def train_lstm(X_train, y_train, X_val, y_val, units, optimizer, epochs, batch_size):
    """
    Entrena un modelo LSTM con los parámetros especificados.
    """
    model = Sequential([
        LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def evaluate_configurations(configurations, X_train, y_train, X_val, y_val, X_test, y_test, output_path):
    """
    Evalúa todas las configuraciones y guarda los resultados en un archivo CSV.
    """
    results = []

    for config in configurations:
        logging.info(f"Evaluando configuración: {config}")
        model = train_lstm(
            X_train, y_train, X_val, y_val,
            units=config['units'],
            optimizer=config['optimizer'],
            epochs=config['epochs'],
            batch_size=config['batch_size']
        )
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        results.append({**config, **metrics})

    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_path, "evaluation_results.csv")
    results_df.to_csv(results_file, index=False)
    logging.info(f"Resultados guardados en {results_file}")
    return results_df


def plot_comparative_results(results_df, output_path):
    """
    Genera gráficos comparativos de las métricas para las configuraciones evaluadas.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(results_df.index, results_df['rmse'], color='skyblue')
    axes[0].set_title('RMSE por Configuración')
    axes[0].set_xlabel('Configuración')
    axes[0].set_ylabel('RMSE')

    axes[1].bar(results_df.index, results_df['r2'], color='lightgreen')
    axes[1].set_title('R² por Configuración')
    axes[1].set_xlabel('Configuración')
    axes[1].set_ylabel('R²')

    plt.tight_layout()
    plot_file = os.path.join(output_path, "comparative_results.png")
    plt.savefig(plot_file)
    logging.info(f"Gráficos comparativos guardados en {plot_file}")
    plt.show()


def main():
    # Configuración de parámetros y rutas
    output_path = "./"
    os.makedirs(output_path, exist_ok=True)

    # Datos de ejemplo (reemplazar con carga real de datos)
    X_train = np.random.rand(100, 10, 3)
    y_train = np.random.rand(100, 1)
    X_val = np.random.rand(20, 10, 3)
    y_val = np.random.rand(20, 1)
    X_test = np.random.rand(20, 10, 3)
    y_test = np.random.rand(20, 1)

    # Definir configuraciones a evaluar
    configurations = [
        {'units': 50, 'optimizer': Adam(), 'epochs': 10, 'batch_size': 16},
        {'units': 100, 'optimizer': Adam(), 'epochs': 20, 'batch_size': 32},
        {'units': 50, 'optimizer': SGD(), 'epochs': 15, 'batch_size': 16}
    ]

    # Evaluar configuraciones y guardar resultados
    results_df = evaluate_configurations(
        configurations, X_train, y_train, X_val, y_val, X_test, y_test, output_path)

    # Visualizar resultados
    plot_comparative_results(results_df, output_path)


if __name__ == "__main__":
    main()
