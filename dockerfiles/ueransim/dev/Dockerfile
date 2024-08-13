FROM ubuntu:20.04

WORKDIR /ueransim

ENV DEBIAN_FRONTEND noninteractive

# Install runtime dependencies + debug tools
RUN apt update \
    && apt install -y git gcc libc-dev libsctp-dev lksctp-tools wget vim iperf3 \
    && apt install -y net-tools tcpdump curl libmnl0 libyaml-0-2 iproute2 iptables tcpdump iputils-ping \
    && apt-get clean \
    && wget https://go.dev/dl/go1.19.linux-amd64.tar.gz -O go_installer.tar.gz \
    && tar -C /usr/local -xzf go_installer.tar.gz \
    && rm go_installer.tar.gz \
    && wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1-linux-x86_64.sh -O cmake_installer.sh \
    && chmod +x cmake_installer.sh \
    && ./cmake_installer.sh --skip-license \
    && git clone -b v3.2.6 -j `nproc` https://github.com/aligungr/UERANSIM \
    && cd ./UERANSIM \
    && make

# Configurar el PATH para Go
ENV PATH="/usr/local/go/bin:${PATH}"

# Instalar dependencias de Go
RUN go install github.com/air-verse/air@latest

# Configurar el PATH de dependencias de go
ENV PATH="/root/go/bin:${PATH}"

COPY ../tun/* ./
RUN chmod +x init.sh

VOLUME [ "/ueransim/config" ]
