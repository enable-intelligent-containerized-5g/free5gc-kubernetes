import time, joblib, os, json, csv, sys, math
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
from keras.regularizers import l2

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

def plot_results(y_test_invertido, y_pred_invertido, name, large_name, model, model_type, training_time, base_name, info_models_path, info_models_path_csv, time_steps):
    # Evaluate the model
    mse = math.sqrt(mean_squared_error(y_test_invertido, y_pred_invertido))
    r2 = r2_score(y_test_invertido, y_pred_invertido)
    print(f'RMSE: {mse:.4f}, R²: {r2:.4f}')
        
    # Evaluate the model: MSE and R² for each output (CPU, Memory and Throughput)
    mse_cpu = math.sqrt(mean_squared_error(y_test_invertido[:, 0], y_pred_invertido[:, 0]))  # For CPU
    mse_mem = math.sqrt(mean_squared_error(y_test_invertido[:, 1], y_pred_invertido[:, 1]))  # For Memory
    mse_thrpt = math.sqrt(mean_squared_error(y_test_invertido[:, 2], y_pred_invertido[:, 2]))  # For Throughput
    r2_cpu = r2_score(y_test_invertido[:, 0], y_pred_invertido[:, 0])  # R² para CPU
    r2_mem = r2_score(y_test_invertido[:, 1], y_pred_invertido[:, 1])  # R² para Memoria
    r2_thrpt = r2_score(y_test_invertido[:, 2], y_pred_invertido[:, 2])  # R² para Throughput
    
    print(f'CPU -> R²: {r2_cpu:.4f}, RMSE: {mse_cpu:.4f}')
    print(f'Memory -> R²: {r2_mem:.4f}, RMSE: {mse_mem:.4f}')
    print(f'Throughput -> R²: {r2_thrpt:.4f}, RMSE: {mse_thrpt:.4f}')
    
    
    # Create the figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

    # CPU Graph
    ax1.scatter(y_test_invertido[:, 0], y_pred_invertido[:, 0], color='blue', label='Prediction vs Real CPU')
    ax1.plot([min(y_test_invertido[:, 0]), max(y_test_invertido[:, 0])], 
            [min(y_test_invertido[:, 0]), max(y_test_invertido[:, 0])], color='red', linestyle='--', label='CPU reference line')
    ax1.set_xlabel('Real CPU Usage')
    ax1.set_ylabel('Predicted CPU Usage')
    ax1.set_title(f'CPU Predictions (RMSE: {mse_cpu:.4f}, R²: {r2_cpu:.4f})')
    ax1.legend()
    ax1.grid(True)
    # Memory graph
    ax2.scatter(y_test_invertido[:, 1], y_pred_invertido[:, 1], color='green', label='Prediction vs Real Memory')
    ax2.plot([min(y_test_invertido[:, 1]), max(y_test_invertido[:, 1])], 
            [min(y_test_invertido[:, 1]), max(y_test_invertido[:, 1])], color='orange', linestyle='--', label='Memory reference line')
    ax2.set_xlabel('Real Memory Usage')
    ax2.set_ylabel('Predicted Memory Usage')
    ax2.set_title(f'Memory Predictions (RMSE: {mse_mem:.4f}, R²: {r2_mem:.4f})')
    ax2.legend()
    ax2.grid(True)
    # Throughput graph
    ax3.scatter(y_test_invertido[:, 2], y_pred_invertido[:, 2], color='green', label='Prediction vs Real Throughput')
    ax3.plot([min(y_test_invertido[:, 2]), max(y_test_invertido[:, 2])], 
            [min(y_test_invertido[:, 2]), max(y_test_invertido[:, 2])], color='blue', linestyle='--', label='Throughput reference line')
    ax3.set_xlabel('Real Throughput Usage')
    ax3.set_ylabel('Predicted Throughput')
    ax3.set_title(f'Throughput Predictions (MSE: {mse_thrpt:.4f}, R²: {r2_thrpt:.4f})')
    ax3.legend()
    ax3.grid(True)

    # Title
    fig.suptitle(f'{large_name} ({name}) model\nMSE: {mse:.4f}, R²: {r2:.4f}', fontsize=14)
    # Adjust the graphs
    plt.tight_layout(pad=0.8) 
    # Show plot
    # plt.show()
    
    # Save plot
    fig_path = "figures/"
    fig_format = "png"
    full_name, fig_uri = save_figure(plt, fig_path, name, fig_format, base_name)
    
    # Save model
    models_path = "saved_models/"
    model_uri, size = save_model_switch(full_name, model, model_type, models_path)
    
    # Save info
    if model_uri != "none":
        # Save info
        new_model = {
            'name': name,
            'uri': model_uri,
            'size': size,
            'figure': fig_uri,
            'r2':r2,
            'mse': mse,
            'r2-cpu':r2_cpu,
            'r2-mem':r2_mem,
            'r2-thrpt':r2_thrpt,
            'mse-cpu': mse_cpu,
            'mse-mem': mse_mem,
            'mse-thrpt': mse_thrpt,
            'training-time': training_time,
            'time-step' : time_steps,
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

def ml_model_training(directory_path, dataset_name, dataset_ext, info_models_path, info_models_path_csv, cpu_column, mem_column, thrpt_column, time_steps, base_name):
    ##################################################################
    ###                   Common configuration                     ###
    ##################################################################
    
    data_path = f"{directory_path}{dataset_name}.{dataset_ext}"
    
    # Load data from a CSV file
    def load_data_from_csv(csv_file):
        data = pd.read_csv(csv_file)
        return data

    # Load dataset from a CSV file
    df = load_data_from_csv(data_path)
    
    # We select the columns that we are going to use for the prediction
    data_values = df[[cpu_column, mem_column, thrpt_column]].values
    # Scale the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_values) # Comun Dataset
    
    # Create secuences
    def create_sequences_multivariate(data, time_steps):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:i + time_steps]) 
            y.append(data[i + time_steps]) 
        return np.array(X), np.array(y)
    
    X, y = create_sequences_multivariate(data_scaled, time_steps)
    
    
    
    ##################################################################
    ###                         LSTM, GRU                          ###
    ##################################################################

    if False :
        # Split data
        train_size = int(len(X) * 0.7)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Define the LSTM model
        lstm_model = Sequential()
        lstm_model.add(LSTM(64, return_sequences=True, input_shape=(time_steps, X.shape[2])))
        lstm_model.add(LSTM(32, return_sequences=False))
        # lstm_model.add(LSTM(50))
        lstm_model.add(Dense(3))
        lstm_model.compile(optimizer='adam', loss='mse')
        #  Train the model
        start_time = time.time()
        history = lstm_model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
        end_time = time.time()
        training_time_lstm = end_time - start_time
        
        # Defining the GRU model
        gru_model = Sequential()
        gru_model.add(GRU(128, return_sequences=True, input_shape=(time_steps, X.shape[2])))
        gru_model.add(GRU(64))
        gru_model.add(Dense(3)) 
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
            plot_results(y_test_invertido, y_pred_invertido, name, large_name, model, model_type, training_time, base_name, info_models_path, info_models_path_csv, time_steps)

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
    ###             XGBRegressor, RandomForestRegressor            ###
    ###                    and LinearRegression                    ###
    ##################################################################

    if True :
        X_train, X_test, y_train, y_test = train_test_split(X.reshape(X.shape[0], -1), y, test_size=0.3, random_state=42)

        # Create the models
        xgb_model = XGBRegressor(n_estimators=100, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        # dt_model = DecisionTreeRegressor(random_state=42)
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
        
        # start_time = time.time()
        # dt_model.fit(X_train, y_train)
        # end_time = time.time()
        # training_time_dt = end_time - start_time
        
        start_time = time.time()
        lr_model.fit(X_train, y_train)
        end_time = time.time()
        training_time_lr = end_time - start_time

        # Evaluate the models
        for model, name, large_name, model_type, training_time in zip([xgb_model, rf_model, lr_model], ['XGBoost', 'RF', 'LR'], ['Extreme Gradient Boosting', 'Random Forest', 'Linear Regression'], ['xgboost', 'sklearn', 'sklearn'], [training_time_xgb, training_time_rf, training_time_lr]):
            print()
            print(f"MODEL: {large_name}")

            # Make the predictions
            y_pred = model.predict(X_test)
            # Invert the normalization to obtain the original values
            y_pred_invertido = scaler.inverse_transform(y_pred)
            y_test_invertido = scaler.inverse_transform(y_test)
            
            # Plot
            plot_results(y_test_invertido, y_pred_invertido, name, large_name, model, model_type, training_time, base_name, info_models_path, info_models_path_csv, time_steps)
            
        
        
    ##################################################################
    ###                    Multilayer Perceptron                   ###
    ##################################################################
            
    # Create the lags features
    if False :
        def create_lagged_features(data, lag):
            X, y = [], []
            for i in range(lag, len(data)):
                X.append(data[i-lag:i].flatten())
                y.append(data[i])
            return np.array(X), np.array(y)
        
        X, y = create_lagged_features(data_scaled, time_steps)
        
        # Divide the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Define the MLP model
        mlp_model = Sequential()
        mlp_model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
        mlp_model.add(Dense(64, activation='relu'))
        mlp_model.add(Dense(3))  # Salida para predecir tanto CPU como Memoria
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
        # Make tdirectory_pathhe predictions
        y_pred = mlp_model.predict(X_test)
        
        # Invert teh data to get the real values
        y_pred_invertido = scaler.inverse_transform(y_pred)
        y_test_invertido = scaler.inverse_transform(y_test)
        
        # Plot
        plot_results(y_test_invertido, y_pred_invertido, name, large_name, mlp_model, model_type, training_time_mlp, base_name, info_models_path, info_models_path_csv, time_steps)

def main():
    print("Ml Model Training")
    
    if len(sys.argv) != 4:
        print("Usage: python3 train-models.py <directory-path> <dataset-name> <time-steps>")
        sys.exit(1)

    # Parameters
    # print(sys.argv[0], sys.argv[1], sys.argv[2],sys.argv[3])
    directory_path = sys.argv[1]
    dataset_name = sys.argv[2]
    # from_step = int(sys.argv[3])
    time_steps = int(sys.argv[3])
    
    # Params
    dataset_extension = "csv"
    cpu_column = "cpu-average"
    mem_column = "mem-average"
    thrpt_column = "throughput-average"
    base_name =  f"{dataset_name}_total-steps-{time_steps}"
    info_models_path = f"models-info/models_info_{base_name}.json"
    info_models_path_csv = f"models-info/models_info_{base_name}.csv"
    for file_path in [info_models_path, info_models_path_csv]:
        if os.path.exists(file_path):  # Verifica si el archivo existe
            os.remove(file_path)      # Elimina el archivo
            print(f"File deleted: {file_path}")
        else:
            print(f"File dont'n deleted: {file_path}")
    
    
    for i in range(time_steps):
        current_time_steps = i+1
        base_name_full =  f"{dataset_name}_total-steps-{current_time_steps}"
        
        if current_time_steps > 100:
            continue
        
        print(f"\n######## CURRENT TIMESTEP: {current_time_steps} #############")
                
        ml_model_training(directory_path, dataset_name, dataset_extension, info_models_path, info_models_path_csv, cpu_column, mem_column, thrpt_column, current_time_steps, base_name_full)
        
        # Load data
        df = pd.read_csv(info_models_path_csv)
        # Columns
        columns = ['size', 'r2', 'mse', 'r2-cpu', 'r2-mem', 'r2-thrpt', 'mse-cpu', 'mse-mem', 'mse-thrpt', 'training-time']
        # Titles
        titles = ['Size', 'R2', 'MSE', 'R2 CPU', 'R2 Memory', 'R2 Throughput', 'MSE CPU', 'MSE Memory', 'MSE Throughput', 'Training time']
        # Create the plots
        for i, col in enumerate(columns):
            # Crear una figura
            plt.close('all')
            plt.figure(figsize=(10, 6))

            # Graficar la columna específica
            plt.bar(df['name'], df[col], color='skyblue')
            plt.title(titles[i])
            plt.xlabel('Model')
            plt.ylabel(titles[i])

            # Mostrar la gráfica
            # plt.show()
            plt.savefig(f"figures-comparation/figure-comparation_{base_name}_metric-{titles[i]}.png")


if __name__ == "__main__":
    main()
    

    
    
