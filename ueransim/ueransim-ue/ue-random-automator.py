import numpy as np
import matplotlib.pyplot as plt
import sys, time, math, subprocess

import random, sys
import matplotlib.pyplot as plt

def generate_user_records(average, deviation, minutes, max_ues):
    # records = [max(0, int(random.gauss(average, deviation))) for _ in range(minutes)]
    records = [max(0, min(int(random.gauss(average, deviation)), max_ues)) for _ in range(minutes)]
    return records

def plot_records(records, name, scale):
    plt.figure(figsize=(10, 6))
    plt.plot(records, marker='o', color='b', label='Records per minute')
    plt.axhline(y=sum(records)/len(records), color='r', linestyle='--', label='Average')
    plt.title("Sintetic trafic")
    plt.xlabel(f"Time (x{scale}minutes)")
    plt.ylabel("Number of users")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"random_traffic_{name}.png", dpi=300)

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

if len(sys.argv) != 6:
    print("Usage: python3 random-traffic.py <ues-average> <deviation> <duration> <scale-seconds> <max-ues>")
    sys.exit(1)

# Parameters
average = int(sys.argv[1])
deviation = int(sys.argv[2])
duration = int(sys.argv[3])
scale = int(sys.argv[4])
max_ues = int(sys.argv[5])

records = generate_user_records(average, deviation, duration+1, max_ues)
prefix_name = f"{average}_{deviation}_{duration}_{scale}"
plot_records(records, prefix_name, scale/60)

print(f"\n##### UE random automator with UE average: {average} and deviation: {deviation} #####\n")

seconds_general = duration*scale
minutes_g, seconds_g = get_minutes(seconds_general)
print(f"Total time required: {minutes_g}:{seconds_g} minutes\n")

y_size = len(records)

# Variables iniciales
ue_prefix = "automator-ue"
wait_time = (duration*scale)/(y_size-1)
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
for t, y_value in enumerate(records):
    seconds_partial = (t)*scale
    minutes_p, seconds_p = get_minutes(seconds_partial) 
    print(f"\n--------Progress: {percentil(t, y_size-1)}%, Time: {minutes_p}:{seconds_p} minutes. (Wait: {wait_time} seconds)")
    
    y_value = math.ceil(y_value)  # Redondear número de pods
    start_time = time.time()
    
    y_value_dif = abs(y_value_previous - y_value)
    wait_time_sub = wait_time
    if y_value_dif != 0:
        wait_time_sub = wait_time/y_value_dif # |a b c d e|    
    print(f"Dif: {wait_time_sub} seconds")
    
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
