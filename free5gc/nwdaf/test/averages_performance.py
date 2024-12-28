import os
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def graficar_una_grafica(x_values, datos, name, titulos, line_style, color, nombre_eje_x="Eje X", nombre_eje_y="Eje Y", titulo_grafica="Comparación de Secuencias"):
    # Graficar cada secuencia con su etiqueta

    plt.close('all')
    plt.figure(figsize=(8, 4))
    x = np.array(x_values)
    for i, secuencia in enumerate(datos):
        y = np.array(secuencia)

        # Graficar
        plt.plot(x, y, linewidth=2, marker="", color=color[i], solid_capstyle='round', linestyle=line_style[i], label=titulos[i])
        # plt.fill_between(x, 0, y, alpha=0.1) 
        
    # Configurar etiquetas de los ejes
    plt.xlabel(nombre_eje_x)
    plt.ylabel(nombre_eje_y)
    plt.title(titulo_grafica)

    # Añadir una leyenda
    plt.legend()
    
    # Mostrar la gráfica
    plt.grid(True)
    # plt.show()
    plt.savefig(f"parallel-logs/figure_performance_{name}.pdf", bbox_inches='tight', pad_inches=0.05)
       

def extraer_tiempos(directorio):
    tiempos = []
    
    # Recorre todos los archivos en el directorio
    for archivo in os.listdir(directorio):
        # Filtra los archivos que coinciden con el patrón "output_parallel_{numero}.log"
        if archivo.startswith("output_proceso_") and archivo.endswith(".log"):
            ruta_archivo = os.path.join(directorio, archivo)
            
            try:
                # Abre el archivo para leer
                with open(ruta_archivo, 'r') as file:
                    lineas = file.readlines()
                    
                    # Busca la línea que contiene "Average time: " y extrae el número
                    for linea in lineas:
                        if "Average time:" in linea:
                            # Usa una expresión regular para extraer el número de la línea
                            tiempo = re.search(r"Average time: (\d+\.\d+) seconds", linea)
                            if tiempo:
                                tiempos.append(float(tiempo.group(1)))  # Agrega el número al listado de tiempos
                            break
            except Exception as e:
                print(f"Error al leer el archivo {archivo}: {e}")
    
    return tiempos

def calcular_promedio(tiempos):
    if tiempos:
        return sum(tiempos) / len(tiempos)
    else:
        return None

# Directorio donde están los archivos .log
directorio = "parallel-logs/parallel_100_s_1"
# Extraer los tiempos de todos los archivos
tiempos = extraer_tiempos(directorio)
# Calcular el promedio de los tiempos
promedio = calcular_promedio(tiempos)

if promedio is not None:
    print(f"El tiempo promedio de s {len(tiempos)} es: {promedio} segundos.")
else:
    print("No se encontraron tiempos para calcular el promedio.")

# Directorio donde están los archivos .log
directorio = "parallel-logs/parallel_500_s_1"
# Extraer los tiempos de todos los archivos
tiempos = extraer_tiempos(directorio)
# Calcular el promedio de los tiempos
promedio = calcular_promedio(tiempos)

if promedio is not None:
    print(f"El tiempo promedio de s {len(tiempos)} es: {promedio} segundos.")
else:
    print("No se encontraron tiempos para calcular el promedio.")




# Directorio donde están los archivos .log
directorio = "parallel-logs/parallel_1_p_1"
# Extraer los tiempos de todos los archivos
tiempos = extraer_tiempos(directorio)
# Calcular el promedio de los tiempos
promedio = calcular_promedio(tiempos)

if promedio is not None:
    print(f"El tiempo promedio de p {len(tiempos)} es: {promedio} segundos.")
else:
    print("No se encontraron tiempos para calcular el promedio.")

# Directorio donde están los archivos .log
directorio = "parallel-logs/parallel_5_p_1"
# Extraer los tiempos de todos los archivos
tiempos = extraer_tiempos(directorio)
# Calcular el promedio de los tiempos
promedio = calcular_promedio(tiempos)

if promedio is not None:
    print(f"El tiempo promedio de p {len(tiempos)} es: {promedio} segundos.")
else:
    print("No se encontraron tiempos para calcular el promedio.")





# Directorio donde están los archivos .log
directorio = "parallel-logs/parallel_1_t_1"
# Extraer los tiempos de todos los archivos
tiempos = extraer_tiempos(directorio)
# Calcular el promedio de los tiempos
promedio = calcular_promedio(tiempos)

if promedio is not None:
    print(f"El tiempo promedio de t {len(tiempos)} es: {promedio} segundos.")
else:
    print("No se encontraron tiempos para calcular el promedio.")

# Directorio donde están los archivos .log
directorio = "parallel-logs/parallel_5_t_1"
# Extraer los tiempos de todos los archivos
tiempos = extraer_tiempos(directorio)
# Calcular el promedio de los tiempos
promedio = calcular_promedio(tiempos)

if promedio is not None:
    print(f"El tiempo promedio de t {len(tiempos)} es: {promedio} segundos.")
else:
    print("No se encontraron tiempos para calcular el promedio.")





x_values = [0, 15, 30, 45, 60, 75, 90, 105]
line_style = ["--", "-", "--", "-", "--", "-"]
color = ["blue", "blue", "red", "red"]

t1_cpu = [0.00465, 0, 3.29, 3.76, 3.79, 0.00391, 0.00369, 0.00364] # 2024-12-17 17:45:00 -> 2024-12-17 17:46:30
t1_mem = [3.13, 3.13, 3.51, 3.88, 4.26, 4.64, 4.64, 4.64]
t5_cpu = [0.000695, 3.24, 27.2, 36.6, 36.5, 0, 0, 0] # 2024-12-17 17:29:30 -> 024-12-17 17:31:15
t5_mem= [1.45, 1.87, 14.0, 14.7, 15.4, 4, 4, 3.91]

# Lista con las secuencias
datos = [t5_cpu, t5_mem, t1_cpu, t1_mem]
# x_values = [x_value, x_value, x_value, x_value]

# Etiquetas para cada secuencia
titulos = ["CPU (5 request)", "Memory (5 request)", "CPU (1 request)", "Memory (1 request)"]
name = "training"

# Llamada a la función
graficar_una_grafica(x_values, datos, name, titulos, line_style, color, nombre_eje_x="Time (seconds)", nombre_eje_y="Usage (%)", titulo_grafica="Model training performance")







# s1_cpu = [0.000251, 0, 0.000376, 0.000437, 0.0154, 0.0103, 0.0131, 0.000377, 0] # 2024-12-17 18:38:00 -> 2024-12-17 18:40:00
# s1_mem = [11.6, 11.6, 11.6, 11.6, 11.3, 11.1, 11.0, 10.7]
s100_cpu = [0.000876, 0.00131, 0.000397, 0.435, 0.706, 0.538, 0.187, 0.0] # 2024-12-17 18:55:30 -> 2024-12-17 18:57:15
s100_mem= [11.1, 11.1, 11.1, 11.8, 12.2, 12.3, 12.6, 12.4]
s500_cpu = [0.000684, 0.730, 1.12, 3.04, 2.44, 2.03, 0.000629, 0.000343] # 2024-12-17 18:45:45 -> 2024-12-17 18:47:30
s500_mem= [11.6, 13.5, 16.4, 13.0, 13.0, 13.0, 13.0, 13.0]

# Lista con las secuencias
datos = [s500_cpu, s500_mem, s100_cpu, s100_mem]
# x_values = [x_value, x_value, x_value, x_value]

# Etiquetas para cada secuencia
titulos = ["CPU (500 request)", "Memory (500 request)", "CPU (100 request)", "Memory (100 request)", "CPU (1 request)", "Memory (1 request)"]
name = "statistics"

# Llamada a la función
graficar_una_grafica(x_values, datos, name, titulos, line_style, color, nombre_eje_x="Time (seconds)", nombre_eje_y="Usage (%)", titulo_grafica="Statistics performance")






p1_cpu = [0.00425, 0, 1.61, 1.75, 1.78, 0, 0, 0.000467] # 2024-12-17 18:00:45 -> 2024-12-17 18:02:20
p1_mem = [20.2, 19.8, 19.8, 19.8, 19.9, 19.9, 19.9, 19.8]
p5_cpu = [0.00738, 0.00843, 0.742, 1.1, 18.5, 13.8, 0.000374, 0.000374] # 2024-12-17 18:03:45 -> 2024-12-17 18:05:30
p5_mem= [19.7, 21.6, 22.2, 22.4, 21.9, 20.4, 20.4, 20.4]

# Lista con las secuencias
datos = [p5_cpu, p5_mem, p1_cpu, p1_mem]
# x_values = [x_value, x_value, x_value, x_value]

# Etiquetas para cada secuencia
titulos = ["CPU (5 request)", "Memory (5 request)", "CPU (1 request)", "Memory (1 request)", "CPU (1 request)", "Memory (1 request)"]
name = "predictions"

# Llamada a la función
graficar_una_grafica(x_values, datos, name, titulos, line_style, color, nombre_eje_x="Time (seconds)", nombre_eje_y="Usage (%)", titulo_grafica="Prediction performance")