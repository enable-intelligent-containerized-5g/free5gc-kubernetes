import numpy as np
import matplotlib.pyplot as plt
import sys, time, math, subprocess

def percentil(a, b):
    return round(a/b*100, 2)

def execute_wait_time(start, end, wait_time):
    execution_time = end - start  # Get execution time
    adjusted_wait_time = wait_time - execution_time  # Adjusted time
    if adjusted_wait_time > 0:
        time.sleep(adjusted_wait_time)  # Wait the adjusted time
    else:
        print(f"Warning: Execution time ({execution_time}) exceeds timeout ({wait_time}).")

def get_minutes (seconds_total):
    minutes = round(seconds_total // 60)
    seconds = round(seconds_total % 60)
    
    return minutes, seconds

def plot_graph(x, y, scale, name):
    plt.figure(figsize=(10, 6)) # Initialize the figure
    
    # Personalizar la gráfica
    plt.plot(x, y, label=f"p2={p2}", marker="o")
    plt.title("Sintetic trafic", fontsize=14)
    plt.xlabel(f"Time (x{scale/60} Minutes)", fontsize=12)
    plt.ylabel("Number of UEs", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Mostrar la gráfica
    # plt.show()
    plt.savefig(f"sintetic_trafic_{name}.png", dpi=300)  
    

if len(sys.argv) != 4:
    print("Use: python3 run-ues.py <num-ues> <period> <time>")
    sys.exit(1)

# Params
n  = int(sys.argv[1])
a1 = n/4 # Amplitude 1
a2 = n/4 # Amplitude 2
p1 = 7 # Period 1
p2 = int(sys.argv[2]) # Period 2 
total_time = int(sys.argv[3])
# p3 = 7 # Period 3
periods = [p2]
mcm = abs(p1*p2) // math.gcd(p1, p2)
mcm = total_time

# Function
points = 10
scale = 180 # In seconds

# mnc = min(mcm_list)
points_total = mcm*points

# Time values
x = np.linspace(0, mcm, mcm*points)
# Calulate the sintetic trafic
y = a1 * np.sin((x * 2 * np.pi) / p1) + a2 * \
    np.sin((x * 2 * np.pi) / p2) + (a1+a2)

seconds_total = len(x)*scale/points
minutes, seconds = get_minutes(seconds_total) 
print(f"Time required: {minutes}:{seconds} minutes\n")
    
y_size = len(y)

prefix_name = f"ue-{n}_p1-{p1}_p2-{p2}_scale-{scale}_time-{total_time}"
plot_graph(x, y, scale, prefix_name)

# Variables iniciales
ue_prefix = "automator-ue"
y_value_previous = 0 

# Apply the resources
try:
    subprocess.run(["kubectl", "apply", "-k", "resources",
                    "-n", "free5gc"], check=True)
except subprocess.CalledProcessError as e:
    print("Error al aplicar la configuración inicial de Kubernetes:", e)
    sys.exit(1)
    
init_state = 0

# Loop for Y values
for t, y_value in enumerate(y):
    wait_time = (points_total*scale)/(points*y_size-1)
    
    seconds_partial = x[t]*scale
    minutes_p, seconds_p = get_minutes(seconds_partial) 
    
    y_value = math.ceil(y_value)  # Redondear número de pods
    start_time = time.time()
    
    y_value_dif = abs(y_value_previous - y_value)
    wait_time_sub = wait_time
    if y_value_dif != 0:
        wait_time_sub = wait_time/y_value_dif # |a b c d e|
    if init_state == 0:  # En el  ciclo, crear los pods iniciales
        wait_time_sub = 0.05
        wait_time = wait_time_sub*y_value_dif 
        
    print(f"\n--------Progress: {percentil(t, y_size-1)}%, Time: {minutes_p}:{seconds_p} minutes. (Wait: {wait_time} seconds)")
    
    if init_state == 0:  # En el  ciclo, crear los pods iniciales
        wait_time_sub = 0
        wait_time = wait_time_sub*y_value_dif 
        print(f"Dif: {wait_time_sub} seconds")
            
        print(f"Initializing with {y_value} pods...")
        for ue in range(1, y_value + 1):
            start_time2 = time.time()
            
            actual_user = ue_prefix+f"{ue}"
            comand = ["kubectl", "apply", "-k",
                        actual_user, "-n", "free5gc"]
            try:
                subprocess.run(comand, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error applying pod {actual_user}:", e)
            
            end_time2 = time.time()
            execute_wait_time(start_time2, end_time2, wait_time_sub)
        
        init_state = 1                
        
    else:
        print(f"Dif: {wait_time_sub} seconds")
        if y_value > y_value_previous:  # Escalar hacia arriba
            print(f"Scaling from {y_value_previous} to {y_value} pods...")
            for ue in range(y_value_previous + 1, y_value + 1):
                start_time2 = time.time()
                
                actual_user = ue_prefix + f"{ue}"
                comand = ["kubectl", "apply", "-k",
                            actual_user, "-n", "free5gc"]
                try:
                    subprocess.run(comand, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error creating pod {actual_user}:", e)
                
                end_time2 = time.time()
                execute_wait_time(start_time2, end_time2, wait_time_sub)
                    
        elif y_value < y_value_previous:  # Escalar hacia abajo
            print(f"Scaling from {y_value_previous} to {y_value} pods...")
            for ue in range(y_value + 1, y_value_previous + 1):
                start_time2 = time.time()
                
                actual_user = ue_prefix + f"{ue}"
                comand = ["kubectl", "delete", "-k",
                            actual_user, "-n", "free5gc"]
                try:
                    subprocess.run(comand, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error deleting pod {actual_user}:", e)
                    
                end_time2 = time.time()
                execute_wait_time(start_time2, end_time2, wait_time_sub)
            
        else:
            print(f"Scaling from {y_value_previous} to {y_value} pods...")


    end_time = time.time()  # Save the end time
    execute_wait_time(start_time, end_time, wait_time)
            
    y_value_previous = y_value
