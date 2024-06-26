import subprocess
import re
import socket
from socket import gethostbyname
from scapy.all import IP, sniff

class Connection:
    def __init__(self, src_ip, src_p, dst_ip, dst_p):
        self.src_ip = src_ip
        self.src_p = src_p
        self.dst_ip = dst_ip
        self.dst_p = dst_p

    def __str__(self):
        return f"Connection de {self.src_ip}:{self.src_p} to {self.dst_ip}:{self.dst_p}"
    

connections = []
        
def resolve_port(service_name, protocol='tcp'):
    try:
        port_number = socket.getservbyname(service_name, protocol)
        return port_number
    except OSError as e:
        # print(f"No se pudo resolver el puerto para el servicio '{service_name}': {e}")
        return service_name

def get_ip_port(cadena):
    # Primero intentamos obtener la IP a partir del posible nombre de dominio
    name_ip_match = re.search(r'^(.*?)\.([^\.]+)$', cadena)  # r'^(.*?)\.(\d+)$'
    dominio_name = name_ip_match.group(1)
    port = name_ip_match.group(2)

    try:
        ip = gethostbyname(dominio_name)
    except:
        # Si no es un nombre de dominio válido o no se puede resolver, asumimos que es una IP directamente
        ip = dominio_name 

    # Ahora extraemos el puerto, que es el fragmento después del último punto
    if port:
        puerto = resolve_port(port) # puerto_match.group(1)
    else:
        return ip, None
    
    return ip, puerto

# Imprimir Result
def print_result(ip, port):
    # Verificar si se obtuvieron valores válidos
    if ip is not None and port is not None:
        print(f"Dirección IP: {ip}")
        print(f"Número de puerto: {port}")
        

# Ejecutar tcpdump y capturar su salida
def run_tcpdump():
    # Comando tcpdump
    command = ["sudo", "tcpdump", "-i", "cni0"]

    # Ejecutar el comando y capturar su salida
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        try:
            for line in iter(proc.stdout.readline, b''):
                # Convertir la salida de bytes a secuencia de bytes

                line = line.strip()
                print(f"tcpdump output: {line.decode('utf-8')}")
                # Extraer la información necesaria de la salida de tcpdump usando expresiones regulares
                src_socket_match = re.search(r"IP\s(\S+)", line.decode('utf-8'))
                dst_socket_match = re.search(r">\s(\S+?):", line.decode('utf-8'))

                # Obtener la dirección IP y el número de puerto
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

                    # Verificar si se obtuvieron valores válidos
                    if ip1 is not None and port1 is not None and ip2 is not None and port2 is not None:
                        connections.append(Connection(ip1, port1, ip2, port2))
                        print_result(ip1, port1)
                        print_result(ip2, port2)
                        for con in connections:
                            print(con)
                        print("----")

        except KeyboardInterrupt:
            print("Terminating tcpdump...")
            proc.terminate()
        except Exception as e:
            print(f"Error running tcpdump: {e}")
            proc.terminate()

if __name__ == "__main__":
    run_tcpdump()
