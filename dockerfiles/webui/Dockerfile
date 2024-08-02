# Build stage for Free5GC
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
    && git clone --recursive -b v1.0.4 -j `nproc` https://github.com/enable-intelligent-containerized-5g/openapi ./openapi \
    && git clone --recursive -b v3.2.5 -j `nproc` https://github.com/enable-intelligent-containerized-5g/free5gc-vanilla ./free5gc \
    && cd free5gc \
    && make webconsole 

# Final stage based on Ubuntu 20.04
FROM ubuntu:20.04
LABEL description="Free5GC v3.2.2"
ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update \
    && apt-get -y install vim net-tools curl libmnl0 libyaml-0-2 iproute2 iptables tcpdump iputils-ping \
    && apt-get clean

WORKDIR /free5gc
RUN mkdir -p config/ log/ public/

COPY --from=builder /go/src/free5gc/webconsole/bin/webconsole ./webui
COPY --from=builder /go/src/free5gc/webconsole/public ./public
COPY --from=builder /go/src/free5gc/config/. ./config/

VOLUME [ "/free5gc/config"]

# Expose necessary ports
EXPOSE 8000 5000
