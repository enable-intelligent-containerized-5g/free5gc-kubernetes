# Conection model
class Conection:
    def __init__(self, src_ip, src_p, dst_ip, dst_p):
        self.src_ip = src_ip
        self.src_p = src_p
        self.dst_ip = dst_ip
        self.dst_p = dst_p

    def __str__(self):
        return f"Conection from {self.src_ip}:{self.src_p} to {self.dst_ip}:{self.dst_p}"