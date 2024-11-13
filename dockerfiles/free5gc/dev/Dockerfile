FROM ubuntu:20.04


# Set the working directory
WORKDIR /free5gc

RUN apt update \
    && apt install -y git make gcc libc-dev wget vim net-tools curl libmnl0 libyaml-0-2 iproute2 iptables tcpdump iputils-ping \
    && wget https://go.dev/dl/go1.19.linux-amd64.tar.gz -O go_installer.tar.gz \
    # && wget https://dl.google.com/go/go1.22.5.linux-amd64.tar.gz -O go_installer.tar.gz \
    && tar -C /usr/local -xzf go_installer.tar.gz \
    && rm go_installer.tar.gz

# Configurar el PATH para Go
ENV PATH="/usr/local/go/bin:${PATH}"

# Clonar el repositorio openapi
# RUN git clone --recursive -b v1.0.4 -j `nproc` https://github.com/enable-intelligent-containerized-5g/openapi ./openapi

# Instalar dependencias de Go
RUN go install github.com/air-verse/air@v1.52.3

# Configurar el PATH de dependencias de go
ENV PATH="/root/go/bin:${PATH}"
