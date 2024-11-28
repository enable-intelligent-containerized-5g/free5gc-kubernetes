import numpy as np
import matplotlib.pyplot as plt
import sys, time, math, subprocess

if len(sys.argv) != 3:
    print("Use: python3 run-ues.py <num-ues> <period>")
    sys.exit(1)

# Parámetros de la función
n  = int(sys.argv[1])
a1 = n/4  # Amplitud
a2 = n/4
p1 = 7  # Período de la segunda función seno
p2 = int(sys.argv[2]) # 4,8,12,16
periods = [p2]


# Dominio de la función
points = 10
scale = 60 # In seconds
x = np.linspace(0, p1*p2, p1*p2*points)  # Valores de x entre 0 y 50

plt.figure(figsize=(10, 6))

for p in periods:
    # Función para calcular el número de pods
    y = a1 * np.sin((x * 2 * np.pi) / p1) + a2 * \
        np.sin((x * 2 * np.pi) / p) + (a1+a2)

    # Variables iniciales
    ue_prefix = "automator-ue"
    wait_time = scale/points
    y_value_previous = math.ceil(y[0])

    # Aplicar configuración inicial de Kubernetes
    try:
        subprocess.run(["kubectl", "apply", "-k", "resources",
                       "-n", "free5gc"], check=True)
    except subprocess.CalledProcessError as e:
        print("Error al aplicar la configuración inicial de Kubernetes:", e)

    # Loop for Y values
    for t, y_value in enumerate(y):
        y_value = math.ceil(y_value)  # Redondear número de pods
        start_time = time.time()

        if t == 0:  # En el  ciclo, crear los pods iniciales
            print(f"Inicializando con {y_value} pods...")
            # Incluir el límite superior para el primer ciclo
            for ue in range(1, y_value + 1):
                actual_user = ue_prefix + f"{ue}"
                comand = ["kubectl", "apply", "-k",
                          actual_user, "-n", "free5gc"]
                try:
                    subprocess.run(comand, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error al crear pod {actual_user}:", e)
            t += 1  # Incrementa t correctamente
            y_value_previous = y_value
            
        else:
            if y_value > y_value_previous:  # Escalar hacia arriba
                print(f"Escalando de {y_value_previous} a {y_value} pods...")
                for ue in range(y_value_previous + 1, y_value + 1):
                    actual_user = ue_prefix+f"{ue}"
                    comand = ["kubectl", "apply", "-k",
                              actual_user, "-n", "free5gc"]
                    try:
                        subprocess.run(comand, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error al crear pod {actual_user}:", e)
            elif y_value < y_value_previous:  # Escalar hacia abajo
                print(f"Reduciendo de {y_value_previous} a {y_value} pods...")
                for ue in range(y_value + 1, y_value_previous + 1):
                    actual_user = ue_prefix+f"{ue}"
                    comand = ["kubectl", "delete", "-k",
                              actual_user, "-n", "free5gc"]
                    try:
                        subprocess.run(comand, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error al eliminar pod {actual_user}:", e)

            # Actualizar valor anterior y esperar
            y_value_previous = y_value
            
            end_time = time.time()  # Guardar el tiempo de fin
            execution_time = end_time - start_time  # Calcular el tiempo de ejecución

            #  el tiempo de espera para la siguiente ejecución
            adjusted_wait_time = wait_time - execution_time  # Restar el tiempo de ejecución del tiempo de espera
            if adjusted_wait_time > 0:
                time.sleep(adjusted_wait_time)  # Dormir el tiempo ajustado
            else:
                print("Advertencia: El tiempo de ejecución excede el tiempo de espera.")
            
    plt.plot(x, y, linestyle='dashed', marker='o', label=f"p2={p}")
    

# Personalizar la gráfica
plt.title("Sintetic trafic", fontsize=14)
plt.xlabel("DecaMinutes", fontsize=12)
plt.ylabel("Num of UEs", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color="black", linewidth=0.8)
plt.legend(fontsize=12)
plt.tight_layout()

# Mostrar la gráfica
plt.show()
