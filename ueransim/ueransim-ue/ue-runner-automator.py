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
    plt.plot(x, y, linestyle='dashed', marker='o', label=f"p2={p}")
    plt.title("Sintetic trafic", fontsize=14)
    plt.xlabel(f"Minutes (x{scale/60})", fontsize=12)
    plt.ylabel("Num of UEs", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Mostrar la gráfica
    # plt.show()
    plt.savefig(f"sintetic_trafic_{name}.png", dpi=300)  
    

if len(sys.argv) != 3:
    print("Use: python3 run-ues.py <num-ues> <period>")
    sys.exit(1)

# Params
n  = int(sys.argv[1])
a1 = n/4 # Amplitude 1
a2 = n/4 # Amplitude 2
p1 = int(sys.argv[2]) # Period 1
p2 = 5 # Period 2 
# p3 = 7 # Period 3
periods = [p1]
mcm_list = []
for p in periods:
    mcm = abs(p*p2) // math.gcd(p, p2)
    mcm_list.append(mcm)

# Function
points = 5
scale = 600 # In seconds

seconds_general = p1*sum(periods)*scale
minutes_g, seconds_g = get_minutes(seconds_general) 
print(f"Total time required: {minutes_g}:{seconds_g} minutes\n")

for tp, p in enumerate(periods):
    print(f"\n##### ({percentil(tp+1, len(periods))}%) UE runner automator with P2: {p} and {n} Pods #####\n")

    mnc = min(mcm_list)
    points_total = mcm*points
    
    # Time values
    x = np.linspace(0, mcm, mcm*points)
    # Calulate the sintetic trafic
    y = a1 * np.sin((x * 2 * np.pi) / p) + a2 * \
        np.sin((x * 2 * np.pi) / p2) + (a1+a2)

    seconds_total = len(x)*scale/points
    minutes, seconds = get_minutes(seconds_total) 
    print(f"Time required: {minutes}:{seconds} minutes\n")
      
    y_size = len(y)
    
    prefix_name = f"{p}_{p2}_{scale}_{points}"
    plot_graph(x, y, scale, prefix_name)

    # Variables iniciales
    ue_prefix = "automator-ue"
    wait_time = (points_total*scale)/(points*y_size-1)
    y_value_previous = 0 
    print(f"Wait time: {wait_time} seconds\n")

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
        seconds_partial = x[t]*scale
        minutes_p, seconds_p = get_minutes(seconds_partial) 
        print(f"\n--------Progress: {percentil(t, y_size-1)}%, Time: {minutes_p}:{seconds_p} minutes. (Wait: {wait_time} seconds)")
        
        y_value = math.ceil(y_value)  # Redondear número de pods
        start_time = time.time()
        
        y_value_dif = abs(y_value_previous - y_value)
        wait_time_sub = wait_time
        if y_value_dif != 0:
            wait_time_sub = wait_time/y_value_dif # |a b c d e|    
        print("Dif: ", wait_time_sub)
        
        if init_state == 0:  # En el  ciclo, crear los pods iniciales
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
