FROM golang:1.19rc1-stretch AS builder

# Rename repo after Debian bulls-eye release
RUN sed -i 's/stable\/updates/stable-security\/updates/' /etc/apt/sources.list \
    && sed -i s/deb.debian.org/archive.debian.org/g /etc/apt/sources.list \
    && sed -i 's|security.debian.org|archive.debian.org|g' /etc/apt/sources.list \
    && sed -i '/stretch-updates/d' /etc/apt/sources.list

# Install dependencies for building Free5GC
RUN apt-get update \
    && apt-get -y install gcc cmake autoconf libtool pkg-config libmnl-dev libyaml-dev apt-transport-https ca-certificates \
    && curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - \
    && echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y nodejs yarn \
    && apt-get clean

# Get and build Free5GC
RUN cd $GOPATH/src \
    && git clone --recursive -b v3.2.01 -j `nproc` https://github.com/enable-intelligent-containerized-5g/free5gc-vanilla ./free5gc \
    && cd free5gc \
    # nwdaf-mtlf, nwdaf-anlf
    && make nwdaf-mtlf

# Build stage for NWDAF
# python:3.7.7-stretch
FROM python:3.7.16-slim-buster 
LABEL description="NWDAF v3.0.5"
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /free5gc

# Copy files from builder
RUN mkdir -p config/ log/ data/ models/ ml_model_training_scripts/

# Copy artifacts from builder stage
COPY --from=builder /go/src/free5gc/bin/* ./
COPY --from=builder /go/src/free5gc/config/. ./config/
COPY --from=builder /go/src/free5gc/NFs/nwdaf-mtlf/data/* ./data/
COPY --from=builder /go/src/free5gc/NFs/nwdaf-mtlf/ml_model_training_scripts/* ./ml_model_training_scripts/

# Install dependencies for building Free5GC
RUN apt-get update \
    && apt-get -y install wget libtool libmnl-dev libyaml-dev apt-transport-https ca-certificates git vim net-tools curl libmnl0 libyaml-0-2 iproute2 iptables tcpdump iputils-ping \
    && pip install pandas==v1.3.5 \
    && pip install joblib==1.3.2 \
    && pip install matplotlib==v3.5.3 \
    && pip install scikit-learn==v1.0.2 \
    # Go package
    && wget https://go.dev/dl/go1.19.linux-amd64.tar.gz -O go1.19_installer.tar.gz \
    && tar -C /usr/local -xzf go1.19_installer.tar.gz \
    && rm go1.19_installer.tar.gz

# Expose necessary ports
EXPOSE 8000 5000