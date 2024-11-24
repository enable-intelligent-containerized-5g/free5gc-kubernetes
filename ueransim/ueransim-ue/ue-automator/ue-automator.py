import os, yaml, sys, shutil

# Validate a positive integer
def validate_num(num, name):
    if not num.isdigit() or int(num) <= 0:
        print(f"Error: <{name}> must be a positive integer.")
        sys.exit(1)

# Replace the map_values in the output_file
def replace_map_values(output_file, replacements):
    if not os.path.isfile(output_file):
        print(f"Error: No found the '{output_file}' file.")
        sys.exit(1)

    # Iterar sobre las claves y valores del diccionario
    with open(output_file, 'r') as file:
        content = file.read()

    for key, value in replacements.items():
        content = content.replace(f"{key}", f"{value}")

    # Escribir los cambios de vuelta al archivo
    with open(output_file, 'w') as file:
        file.write(content)
        
def validate_files(source_path, files):
    for file in files:
        file_path = os.path.join(source_path, file)
        if not os.path.isfile(file_path):
            print(f"Error: The input file '{file_path}' no found.")
            sys.exit(1)
            
def replace_values(source_path, output_path, file, ue_folder, map_values):
    # Crear el archivo dentro de la carpeta
    shutil.copy(os.path.join(source_path, file), os.path.join(ue_folder, file))

    # Reemplazar las claves del mapa con sus valores en el archivo
    replace_map_values(os.path.join(ue_folder,file), map_values)

# Verify the params
if len(sys.argv) != 4:
    print("Use: python3 ue-automator.py <start_ue> <end_ue> <slice#>")
    sys.exit(1)
# Params
start_ue = sys.argv[1]
end_ue = sys.argv[2]
slice = sys.argv[3]
# Files
source_path = "base/"
output_path = "../"
ue_config_file = "ue.yaml"
ue_deployment_file = "ue-deployment.yaml"
ue_wrapper_file = "wrapper.sh"
ue_kustomization_file = "kustomization.yaml"
files_to_replace = [ue_config_file, ue_deployment_file, ue_wrapper_file, ue_kustomization_file]

if slice == "slice1": # Slice 1
    source_path = "base1"
elif slice == "slice2": # Slice 2
    source_path = "base1"
else: # Both slices
    source_path = "base3"

# Validate <start_ue> and <end_ue>
validate_num(start_ue, "start_ue")
validate_num(end_ue, "end_ue")
start_ue = int(start_ue)
end_ue = int(end_ue)
ue_prefix = "automator-ue"

if start_ue > end_ue:
    print("Error: <start_ue> must be smaller than <end_ue>.")
    sys.exit(1)
    

# Validate if the file exist
validate_files(source_path, files_to_replace)
    
# Definir un diccionario de reemplazos: clave -> valor
ue_map_values = {
    "CODE_MCC": "001",
    "CODE_MNC": "01",
    "CODE_KEY": "465B5CE8B199B49FAA5F0A2EE238A6BC",
    "CODE_OP": "E8ED289DEBA952E4283B54E88E6183CA",
    "CODE_OP_TYPE": "OPC",
    "CODE_APN_1": "internet",
    "CODE_SST_1": "1",
    "CODE_SD_1": "000001",
    "CODE_APN_2": "streaming",
    "CODE_SST_2": "1",
    "CODE_SD_2": "000002",
}

# Recorrer el rango de números
for i in range(start_ue, end_ue + 1):
    # Crear la carpeta "ue_<número>"
    ue_folder = os.path.join(output_path, f"{ue_prefix}{i}/")
    os.makedirs(ue_folder, exist_ok=True)
    
    # Generar el nuevo SUPI (IMSI)
    contador = f"{i:010d}"  # Rellenar con ceros para que tenga siempre 10 dígitos
    new_supi = f"imsi-{ue_map_values['CODE_MCC']}{ue_map_values['CODE_MNC']}{contador}"

    # Agregar el nuevo SUPI al diccionario
    ue_map_values["CODE_IMSI_UE"] = new_supi
    ue_map_values["CODE_NUM_UE"] = i
    
    # replace values
    for file in files_to_replace:
        # Replace values
        replace_values(source_path, output_path, file, ue_folder, ue_map_values)
    
    # Confirmar progreso
    print(f"Created: {ue_folder} with SUPI = {new_supi}")
    
print(f"¡Created {str(end_ue - start_ue + 1)} UEs!")
    
# Add kustomization
ue_folders = [f"{ue_prefix}{i}" for i in range(start_ue, end_ue + 1)]
kustomiation_path = os.path.join(output_path, "kustomization.yaml")

if os.path.exists(kustomiation_path):
    with open(kustomiation_path, "r") as file:
        kustomization = yaml.safe_load(file)
else:
    # Crear la estructura básica si el archivo no existe
    kustomization = {
        "apiVersion": "kustomize.config.k8s.io/v1beta1",
        "kind": "Kustomization",
        "resources": [
            "resources",
            "default-ue1",
            "default-ue2",
            "default-ue3",
        ]
}

# Filtrar los UEs que ya existen en el archivo kustomization.yaml
existing_ues = set(kustomization.get("resources", []))

# Agregar solo los UEs que no están en la lista
for ue in ue_folders:
    if ue not in existing_ues:
        kustomization["resources"].append(ue)

# Escribir el archivo actualizado
with open(kustomiation_path, "w") as file:
    yaml.dump(kustomization, file, default_flow_style=False)

print(f"¡Added {kustomiation_path} for the UEs!")
