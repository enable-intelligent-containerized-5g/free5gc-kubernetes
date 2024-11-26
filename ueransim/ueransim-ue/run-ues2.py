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
x = np.linspace(0, p1*p2, p1*p2*points)  # Valores de x entre 0 y 50

plt.figure(figsize=(10, 6))
for p in periods:
    y = a1 * np.sin((x * 2 * np.pi) / p1) + a2 * np.sin((x * 2 * np.pi) / p) +  (a1+a2)
    
    ue_prefix = "automator-ue"
    y_len = len(y)
    t = 0
    y_value_previous = 1
    wait_time = 3
    print(f"Y: {y}")
    result = subprocess.run(["kubectl", "apply", "-k", "resources", "-n", "free5gc"])
    for i in range(0, len(y), points):
        values = y[i:i+points]
        print(f"Values: {values}, LenValues: {len(values)}, i: {i}, i+points-1: {i+points}")
        print(f"Time: {t}")
        print(f"From {math.ceil(values[0])} to {math.ceil(values[-1])}")  
        for j in range(0, len(values)):
            print("Subprocess")
            y_value = math.ceil(values[j])
            if t == 0:
                for ue in range(y_value_previous,y_value+1):
                    actual_user = ue_prefix+f"{ue}"
                    comand = ["kubectl", "apply", "-k", actual_user, "-n", "free5gc"]
                    result = subprocess.run(comand)
                    # print(comand)
            else:
                if y_value_previous < y_value: # Increase (Create new UEs)
                    for ue in range(y_value_previous,y_value+1):
                        actual_user = ue_prefix+f"{ue}"
                        comand = ["kubectl", "apply", "-k", actual_user, "-n", "free5gc"]
                        result = subprocess.run(comand)
                        # print(comand)
                    
                elif y_value_previous > y_value: # Decrease (Delete UEs)
                    for ue in range(y_value,y_value_previous+1):
                        actual_user = ue_prefix+f"{ue}"
                        comand = ["kubectl", "delete", "-k", actual_user, "-n", "free5gc"]
                        result = subprocess.run(comand)
                        # print(comand)
                    
                    # print("Salida estándar:")
                    # print(result.stdout)
                    # print("Error estándar:")
                    # print(result.stderr)
                else:
                    print("y_value_previous == y_value")
                    
            y_value_previous = y_value
            
        t =+1
        time.sleep(wait_time)
    
    plt.plot(x, y, linestyle='dashed', marker='o', label=f"p2={p}")
    
    

# Personalizar la gráfica
plt.title("sintetic trafic", fontsize=14)
plt.xlabel("minutes", fontsize=12)
plt.ylabel("users", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.axhline(0, color="black", linewidth=0.8)
plt.legend(fontsize=12)
plt.tight_layout()

# Mostrar la gráfica
plt.show()
