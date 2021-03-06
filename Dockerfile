# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG CUDA_VERSION=11.1
ARG OS_VERSION=18.04
ARG ssh_prv_key
ARG ssh_pub_key

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
LABEL maintainer="NVIDIA CORPORATION"

ENV TRT_VERSION 7.2.2.3
SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    zlib1g-dev \
    git \
    pkg-config \
    sudo \
    ssh \
    libssl-dev \
    pbzip2 \
    pv \
    bzip2 \
    unzip \
    devscripts \
    lintian \
    fakeroot \
    dh-make \
    build-essential

# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install TensorRT
RUN cd /tmp &&\
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb &&\
    dpkg -i nvidia-machine-learning-repo-*.deb && apt-get update
RUN v="${TRT_VERSION%.*}-1+cuda${CUDA_VERSION%.*}" &&\
    apt-get install -y libnvinfer7=${v} libnvinfer-plugin7=${v} libnvparsers7=${v} libnvonnxparsers7=${v} libnvinfer-dev=${v} libnvinfer-plugin-dev=${v} libnvparsers-dev=${v} python3-libnvinfer=${v} &&\
    apt-mark hold libnvinfer7 libnvinfer-plugin7 libnvparsers7 libnvonnxparsers7 libnvinfer-dev libnvinfer-plugin-dev libnvparsers-dev python3-libnvinfer

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
COPY tensorrt/requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install Cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download NGC client
RUN cd /usr/local/bin && wget https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip && unzip ngccli_cat_linux.zip && chmod u+x ngc && rm ngccli_cat_linux.zip ngc.md5 && echo "no-apikey\nascii\n" | ngc config set

# Install pytorch
RUN pip3 install torch==1.9.0.dev20210414+cu111 torchvision==0.10.0.dev20210414+cu111 -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html

# Install pytorch examples & replace NMIST training script to use custom NN
USER trtuser
RUN cd /home/trtuser && git clone https://github.com/pytorch/examples.git
USER trtuser
COPY examples/mnist/main.py /home/trtuser/examples/mnist/main.py

# Train NN on default parameters
RUN cd /home/trtuser/examples/mnist && \
    python3 main.py --save-model --dry-run

# Clone & build tensorrt sources
RUN cd /workspace && \
    git clone -b add-hardshrink https://github.com/mkotyushev/TensorRT.git TensorRT && \
    cd TensorRT && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build && \
    cmake .. -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_OUT_DIR=`pwd`/out && \
    make -j$(nproc)

# Convert model from atomic operators to Hardshrink
COPY convert_hardshrink.py /workspace/TensorRT
RUN cd /workspace/TensorRT && \
    mkdir /wokrspace/TensorRT/data && \
    python3 convert_hardshrink.py -m /home/trtuser/examples/mnist/mnist_cnn.onnx -s data/mnist_with_hardshrink.onnx

# Download MNIST samples data
RUN cd /wokrspace/TensorRT/samples/python/scripts && \
    python3 download_mnist_pgms.py -o /wokrspace/TensorRT/data

# Set environment and working directory
ENV TRT_LIBPATH /usr/lib/x86_64-linux-gnu
ENV TRT_OSSPATH /workspace/TensorRT
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TRT_OSSPATH}/build/out:${TRT_LIBPATH}"
WORKDIR /workspace

USER trtuser
RUN ["/bin/bash"]
