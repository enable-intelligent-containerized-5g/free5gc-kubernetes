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
    && git clone --recursive -b v3.2.5 -j `nproc` https://github.com/enable-intelligent-and-containerized-5g/free5gc-vanilla ./free5gc \
    && cd free5gc \
    && make nwdaf-mtlf

# Build stage for NWDAF
# python:3.7.7-stretch
FROM python:3.7.16-slim-buster 
LABEL description="NWDAF v3.0.5"
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /free5gc

# Copy files from builder
RUN mkdir -p config/ log/ nwdafmtlf/

# Copy artifacts from builder stage
COPY --from=builder /go/src/free5gc/bin/* ./
COPY --from=builder /go/src/free5gc/config/. ./config/
COPY --from=builder /go/src/free5gc/NFs/nwdaf-mtlf/. ./nwdafmtlf/

# Install dependencies for building Free5GC
RUN apt-get update \
    && apt-get -y install wget libtool libmnl-dev libyaml-dev apt-transport-https ca-certificates git vim net-tools curl libmnl0 libyaml-0-2 iproute2 iptables tcpdump iputils-ping \
    # && pip install openssl==v1.1.1f \
    && apt-get -y install libssl-dev \ 
    && pip install jinja2==v2.11.1 \
    && pip install itsdangerous==v1.1.0 \
    && pip install werkzeug==v2.0.0 \
    && pip install gdown==v3.10.2 \
    && pip install matplotlib==v3.5.3 \
    && pip install keras==v2.11.0 \
    && pip install tensorflow \
    && pip install pandas==v1.3.5 \
    && pip install scikit-learn==v1.0.2 \
    && pip install markupsafe==2.0.1 \
    && wget https://go.dev/dl/go1.19.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.19.linux-amd64.tar.gz \
    && cd nwdafmtlf \
    && pip install -r requirements.txt \
    && python3 setup.py install

# Expose necessary ports
EXPOSE 8000 5000 8081