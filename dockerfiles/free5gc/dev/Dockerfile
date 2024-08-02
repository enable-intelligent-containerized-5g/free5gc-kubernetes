# image base golang:1.15.2-alpine3.12
FROM golang:1.19-alpine3.15

# Set the working directory
WORKDIR /free5gc

# Install required packages
RUN apk add --no-cache git make gcc libc-dev

RUN git clone --recursive -b v1.0.4 -j `nproc` https://github.com/enable-intelligent-containerized-5g/openapi ./openapi

# Install go dependencies
RUN go install github.com/air-verse/air@latest