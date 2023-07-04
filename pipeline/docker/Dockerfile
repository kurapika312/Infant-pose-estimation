ARG UBUNTU_VERSION=18.04
# Use the official ubuntu:18.04 image as the parent image
#FROM ubuntu:${UBUNTU_VERSION}
FROM nvidia/opengl:base-ubuntu18.04

############THE PERMISSION DENIED ERROR SOLUTION : https://phoenixnap.com/kb/docker-permission-denied

# Set the working directory to /app
WORKDIR /app

ARG ARCH=amd64

ADD setup-ubuntu.sh /app
ADD sources.list /app

# Tell Ubuntu non-interactive install
ARG DEBIAN_FRONTEND=noninteractive

RUN chmod +x setup-ubuntu.sh 
RUN ./setup-ubuntu.sh ${ARCH}

RUN apt-get install -y ca-certificates
RUN update-ca-certificates
