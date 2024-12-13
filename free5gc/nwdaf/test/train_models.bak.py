import time, datetime, joblib, os, json, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from collections import namedtuple


def save_model_switch(fig_name, model, model_type, models_path):
    if model_type == 'xgboost':
        format = '.json'
        name = f"model_{fig_name}{format}"
        uri = f"{models_path}{name}"
        model.save_model(uri)
        size = os.path.getsize(uri)
        return uri, size
    
    elif model_type == 'sklearn':
        format = '.pkl'
        name = f"model_{fig_name}{format}"
        uri = f"{models_path}{name}"
        joblib.dump(model, uri)
        size = os.path.getsize(uri)
        return uri, size
    
    elif model_type == 'keras':
        format = '.h5'
        name = f"model_{fig_name}{format}"
        uri = f"{models_path}{name}"
        model.save(uri)
        size = os.path.getsize(uri)
        return uri, size
    
    else:
        return "none", 0

def plot_results(y_test_invertido, y_pred_invertido, name, large_name, model, model_type, training_time, time_step, dataset_name):
    # Evaluate the model
    mse = mean_squared_error(y_test_invertido, y_pred_invertido)
    r2 = r2_score(y_test_invertido, y_pred_invertido)
    print(f'MSE: {mse:.4f}, R²: {r2:.4f}')
    
    # Evaluate the model: MSE and R² for each output (CPU and Memory)
    mse_cpu = mean_squared_error(y_test_invertido[:, 0], y_pred_invertido[:, 0])  # Para la columna de CPU
    mse_mem = mean_squared_error(y_test_invertido[:, 1], y_pred_invertido[:, 1])  # Para la columna de Memoria
    r2_cpu = r2_score(y_test_invertido[:, 0], y_pred_invertido[:, 0])  # R² para CPU
    r2_mem = r2_score(y_test_invertido[:, 1], y_pred_invertido[:, 1])  # R² para Memoria
    
    print(f'CPU - R²: {r2_cpu:.4f}, MSE: {mse_cpu:.4f}')
    print(f'Memory - R²: {r2_mem:.4f}, MSE: {mse_mem:.4f}')
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # CPU Graph
    ax1.scatter(y_test_invertido[:, 0], y_pred_invertido[:, 0], color='blue', label='Prediction vs Real CPU')
    ax1.plot([min(y_test_invertido[:, 0]), max(y_test_invertido[:, 0])], 
            [min(y_test_invertido[:, 0]), max(y_test_invertido[:, 0])], color='red', linestyle='--', label='CPU reference line')
    ax1.set_xlabel('Real CPU Usage')
    ax1.set_ylabel('Predicted CPU Usage')
    ax1.set_title(f'CPU Predictions (MSE: {mse_cpu:.4f}, R²: {r2_cpu:.4f})')
    ax1.legend()
    ax1.grid(True)
    # Memory graph
    ax2.scatter(y_test_invertido[:, 1], y_pred_invertido[:, 1], color='green', label='Prediction vs Real Memory')
    ax2.plot([min(y_test_invertido[:, 1]), max(y_test_invertido[:, 1])], 
            [min(y_test_invertido[:, 1]), max(y_test_invertido[:, 1])], color='orange', linestyle='--', label='Memory reference line')
    ax2.set_xlabel('Real Memory Usage')
    ax2.set_ylabel('Predicted Memory Usage')
    ax2.set_title(f'Memory Predictions (MSE: {mse_mem:.4f}, R²: {r2_mem:.4f})')
    ax2.legend()
    ax2.grid(True)

    # Title
    fig.suptitle(f'{large_name} ({name}) model\nMSE: {mse:.4f}, R²: {r2:.4f}', fontsize=14)
    # Adjust the graphs
    plt.tight_layout(pad=0.8) 
    # Show plot
    # plt.show()
    
    # Save plot
    fig_path = "figures/"
    fig_format = "png"
    base_name =  f"{dataset_name}_steps-{time_step}"
    full_name, fig_uri = save_figure(plt, fig_path, name, fig_format, base_name)
    
    # Save model
    models_path = "saved_models/"
    model_uri, size = save_model_switch(full_name, model, model_type, models_path)
    
    # Save info
    info_models_path = f"models_info_{base_name}.json"
    info_models_path_csv = f"models_info_{base_name}.csv"
    if model_uri != "none":
        # Save info
        new_model = {
            'name': name,
            'uri': model_uri,
            'size': size,
            'figure': fig_uri,
            'r2':r2,
            'mse': mse,
            'r2_cpu':r2_cpu,
            'r2_mem':r2_mem,
            'mse_cpu': mse_cpu,
            'mse_mem': mse_mem,
            'training_time': training_time,
        }
        
        try:
            with open(info_models_path, 'r') as json_file:
                models_info = json.load(json_file)
                # models_info = []    
        except:
            models_info = []

        models_info.append(new_model)

        with open(info_models_path, 'w') as json_file:
            json.dump(models_info, json_file, indent=4)
            
        with open(info_models_path_csv, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=models_info[0].keys())
            writer.writeheader()
            writer.writerows(models_info)

    
def save_figure(plot, fig_path, model_name, format, base_name):
    time.sleep(1)
    # current_date = datetime.datetime.now()
    # formated_current_date = current_date.strftime("%Y-%m-%d") + "_" + current_date.strftime("%H-%M-%S") + "_" + str(current_date.microsecond)
    
    full_name = f"{model_name}_{base_name}"
    fig_uri = f"{fig_path}figure_{full_name}.{format}"
    plot.savefig(fig_uri, format=format, bbox_inches='tight')
    
    return full_name, fig_uri


def ml_model_training(dataset_name, dataset_ext, cpu_column, mem_column):
    ##################################################################
    ###                   Common configuration                     ###
    ##################################################################
    
    data_path = f"{dataset_name}.{dataset_ext}"
    
    # Load data from a CSV file
    def load_data_from_csv(csv_file):
        data = pd.read_csv(csv_file)
        return data

    # Load dataset from a CSV file
    df = load_data_from_csv(data_path)
    
    # We select the columns that we are going to use for the prediction
    data_values = df[[cpu_column, mem_column]].values
    # Scale the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_values) # Comun Dataset
    time_steps = 13 # Steps
    
    # Función para crear las secuencias
    def create_sequences_multivariate(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps])  # Seleccionamos las últimas 'time_steps' filas (como secuencia)
            y.append(data[i + time_steps])  # Valores para predecir
        return np.array(X), np.array(y)
    
    X, y = create_sequences_multivariate(data_scaled, time_steps)
    
    
    
    ##################################################################
    ###                         LSTM, GRU                          ###
    ##################################################################

    if True :
        # Dividir los datos en entrenamiento (70%) y prueba (30%) de forma secuencial
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Define the LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(100, return_sequences=True, input_shape=(time_steps, X.shape[2])))
        # gru_model.add(Dropout(0.3))
        lstm_model.add(LSTM(50))
        lstm_model.add(Dense(2))
        lstm_model.compile(optimizer='adam', loss='mse')
        #  Train the model
        start_time = time.time()
        history = lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
        end_time = time.time()
        training_time_lstm = end_time - start_time
        
        # Defining the GRU model
        gru_model = Sequential()
        gru_model.add(GRU(100, return_sequences=True, input_shape=(time_steps, X.shape[3])))
        gru_model.add(GRU(50))
        gru_model.add(Dense(2)) 
        gru_model.compile(optimizer='adam', loss='mse')
        # Train the model
        start_time = time.time()
        history = gru_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
        end_time = time.time()
        training_time_gru = end_time - start_time
        
        #Evaluate the models
        for model, name, large_name, model_type, training_time in zip([lstm_model, gru_model], ['LSTM', 'GRU'], ['Long Short-Term Memory', 'Gated Recurrent Unit'], ['keras', 'keras'], [training_time_lstm, training_time_gru]):
            print()
            print(f"MODEL: {large_name}")

            # Make predictions
            y_pred = model.predict(X_test)
            # Invert the normalization to obtain the original values
            y_pred_invertido = scaler.inverse_transform(y_pred)
            y_test_invertido = scaler.inverse_transform(y_test)
            
            # Plot
            plot_results(y_test_invertido, y_pred_invertido, name, large_name, model, model_type, training_time, time_steps, dataset_name)

            # # Graficar la pérdida durante el entrenamiento
            # plt.figure(figsize=(10, 6))
            # plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
            # plt.plot(history.history['val_loss'], label='Pérdida de Validación')
            # plt.title('Pérdida durante el Entrenamiento')
            # plt.xlabel('Épocas')
            # plt.ylabel('MSE')
            # plt.legend()
            # plt.show()

    
    
    ##################################################################
    ### XGBRegressor, RandomForestRegressor, DecisionTreeRegressor ###
    ###                    and LinearRegression                    ###
    ##################################################################

    if True :
        X_train, X_test, y_train, y_test = train_test_split(X.reshape(X.shape[0], -1), y, test_size=0.3, random_state=42)

        # Create the models
        # model = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
        # objective='reg:squarederror'
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        dt_model = DecisionTreeRegressor(random_state=42)
        lr_model = LinearRegression()

        # Train the model
        start_time = time.time()
        xgb_model.fit(X_train, y_train)
        end_time = time.time()
        training_time_xgb = end_time - start_time
        
        start_time = time.time()
        rf_model.fit(X_train, y_train)
        end_time = time.time()
        training_time_rf = end_time - start_time
        
        start_time = time.time()
        dt_model.fit(X_train, y_train)
        end_time = time.time()
        training_time_dt = end_time - start_time
        
        start_time = time.time()
        lr_model.fit(X_train, y_train)
        end_time = time.time()
        training_time_lr = end_time - start_time

        # Evaluate the models
        for model, name, large_name, model_type, training_time in zip([xgb_model, rf_model, dt_model, lr_model], ['XGBoost', 'RF', 'DT', 'LR'], ['Extreme Gradient Boosting', 'Random Forest', 'Decision Tree', 'Linear Regression'], ['xgboost', 'sklearn', 'sklearn', 'sklearn'], [training_time_xgb, training_time_rf, training_time_dt, training_time_lr]):
            print()
            print(f"MODEL: {large_name}")

            # Make the predictions
            y_pred = model.predict(X_test)
            # Invert the normalization to obtain the original values
            y_pred_invertido = scaler.inverse_transform(y_pred)
            y_test_invertido = scaler.inverse_transform(y_test)
            
            # Plot
            plot_results(y_test_invertido, y_pred_invertido, name, large_name, model, model_type, training_time, time_steps, dataset_name)
            
        
        
    ##################################################################
    ###                    Multilayer Perceptron                   ###
    ##################################################################
            
    # Create the lags features
    if True :
        def create_lagged_features(data, lag):
            X, y = [], []
            for i in range(lag, len(data)):
                X.append(data[i-lag:i].flatten())
                y.append(data[i])
            return np.array(X), np.array(y)
        
        lag = time_steps
        X, y = create_lagged_features(data_scaled, lag)
        
        # Divide the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Define the MLP model
        mlp_model = Sequential()
        mlp_model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        mlp_model.add(Dense(32, activation='relu'))
        mlp_model.add(Dense(2))  # Salida para predecir tanto CPU como Memoria
        mlp_model.compile(optimizer='adam', loss='mse')

        # Train the model
        start_time = time.time()
        history = mlp_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
        end_time = time.time()
        training_time_mlp = end_time - start_time

        print()
        name = "MLP"
        model_type = 'keras'
        large_name = "Multilayer Perceptron"
        print(f"MODEL: {name}")
        # Make the predictions
        y_pred = mlp_model.predict(X_test)
        
        # Invert teh data to get the real values
        y_pred_invertido = scaler.inverse_transform(y_pred)
        y_test_invertido = scaler.inverse_transform(y_test)
        
        # Plot
        plot_results(y_test_invertido, y_pred_invertido, name, large_name, mlp_model, model_type, training_time_mlp, lag, dataset_name)

def main():
    print("Ml Model Training")
    
    # Params
    # dataset_name = "dataset-old"
    # dataset_name = "dataset_NF_LOAD_AMF_60s_1733380080_1733399405_200_300_7_2"
    # dataset_name = "dataset_NF_LOAD_AMF_60s_1733380800_1733416214_150_300_5_4"
    dataset_name = "dataset_NF_LOAD_AMF_60s_1733446200_1733464830_150_300_5_6"
    dataset_extension = "csv"
    cpu_column = "cpu-average"
    mem_column = "mem-average"
        
    ml_model_training(dataset_name, dataset_extension, cpu_column, mem_column)

if __name__ == "__main__":
    main()
    

    
    
