# Build stage for NWDAF
FROM python:3.7.16-slim-buster 
LABEL description="NWDAF v3.0.5-dev"
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /free5gc

# Install dependencies for building Free5GC
RUN apt-get update \
    && apt-get -y install wget libtool libmnl-dev libyaml-dev apt-transport-https ca-certificates git vim net-tools curl libmnl0 libyaml-0-2 iproute2 iptables tcpdump iputils-ping \
    && pip install pandas==v1.3.5 \
    && pip install joblib==1.3.2 \
    && pip install matplotlib==v3.5.3 \
    && pip install scikit-learn==v1.0.2 \
    # Go package
    && wget https://go.dev/dl/go1.19.linux-amd64.tar.gz -O go_installer.tar.gz \
    && tar -C /usr/local -xzf go_installer.tar.gz \
    && rm go_installer.tar.gz
    # && cd nwdafanlf \
    # && pip install -r requirements.txt \
    # && python3 setup.py install

# Configurar el PATH para Go
ENV PATH="/usr/local/go/bin:${PATH}"

# Install air
RUN go install github.com/air-verse/air@v1.52.3

# Configurar el PATH de dependencias de go
ENV PATH="/root/go/bin:${PATH}"
   
# Expose necessary ports
EXPOSE 8000 5000 8080