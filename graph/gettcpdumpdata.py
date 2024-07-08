import re
import socket
import os
import pickle
from socket import gethostbyname
from Conection import Conection

# Array of conections
conections = []

# Path to the results file
path_tcpdump_results = os.path.join('tcpdumpresults.pkl')

# Path to the tcpdumpdata
# Use: sudo tcpdump -i <interface-name> > path/to/tcpdumpdata.log
path_tcpdump_data = os.path.join('tcpdumpdata.log')

# Save the results
def save_tcpdump_results():
    with open(path_tcpdump_results, 'wb') as file:
        pickle.dump(conections, file)

# Read the results
def upload_tcpdump_results():
    # Deserialize the array of models with pickle
    with open(path_tcpdump_results, 'rb') as file:
        models = pickle.load(file)

    # Print the results
    print_results(models)
        
def resolve_port(service_name, protocol='tcp'):
    try:
        port_number = socket.getservbyname(service_name, protocol)
        return port_number
    except OSError as e:
        # print(f"Can't resolve the '{service_name}': {e}")
        return service_name

def get_ip_port(cadena):
    # Fist we try to get the IP from the domain name
    name_ip_match = re.search(r'^(.*?)\.([^\.]+)$', cadena)  # r'^(.*?)\.(\d+)$'
    if name_ip_match is not None:
        domain_name = name_ip_match.group(1)
        port = name_ip_match.group(2)
    else:
        return None, None

    try:
        ip = gethostbyname(domain_name)
    except:
        # If it is not a valid domain name or cannot be resolved, we assume it is an IP directly
        ip = domain_name 

    # Get the port, it is the last fragment after the last point.
    if port:
        puerto = resolve_port(port)
    else:
        return ip, None
    
    return ip, puerto

# Print the socket
def print_ip_port(message, ip, port):
    if ip is not None and port is not None:
        print(f"{message} > {ip}:{port}")

# Print the result
def print_results(models):
    for model in models:
        print(model)
        

# Read and process the .log file
def read_tcpdump_data():
    # Open the .log file
    try:
        with open(path_tcpdump_data, 'r') as file:
            # Read line by line
            for line in file:
                try:
                    # Process the line
                    line = line.strip()
                    # print(f"tcpdump output: {line}")

                    # Get the necesary information with regulars expresions
                    src_socket_match = re.search(r"IP\s(\S+)", line)
                    dst_socket_match = re.search(r">\s(\S+?):", line)

                    # Get IP an Port
                    if src_socket_match is None:
                        src_socket = None
                    else:
                        src_socket = src_socket_match.group(1)

                    if dst_socket_match is None:
                        dst_socket = None
                    else:
                        dst_socket = dst_socket_match.group(1)

                    if src_socket is not None and dst_socket is not None:
                        ip1, port1 = get_ip_port(src_socket)
                        ip2, port2 = get_ip_port(dst_socket)

                        # Add the new conection to conections array
                        if ip1 is not None and port1 is not None and ip2 is not None and port2 is not None:
                            conections.append(Conection(ip1, port1, ip2, port2))
                            # print_ip_port("src", ip1, port1)
                            # print_ip_port("dst", ip2, port2)
                            # print("----")

                except ValueError:
                                print(f"Can't proccess the line: {line.strip()}")
    except FileNotFoundError:
        print(f"Not found the file {path_tcpdump_data}")
    except PermissionError:
        print(f"Need superuser privileges {path_tcpdump_data}")
    except IsADirectoryError:
        print(f"{path_tcpdump_data} Is a directory, not a file")
    except UnicodeDecodeError:
        print(f"The file {path_tcpdump_data} contains invalid characters")

    # Save the results
    save_tcpdump_results()

    # Print the results
    upload_tcpdump_results()

if __name__ == "__main__":
    read_tcpdump_data()
