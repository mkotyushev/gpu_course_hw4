# Description

This repository contains additional material for YSDA GPU course home assignment 4.

# Files

`examples/mnist/main.py` is script to train CNN with custom Hardshrink activations & save it to ONNX format. Custom layer consist of 10 atomic operators (`Large`, `Where` etc.) implemented in ONNX & TensorRT.

`convert_hardshrink.py` is script to convert ONNX model trained with script above. After conversion, 10 atomic operators of custom layer in model graph are replaced with single `Hardshrink` operator.

`Dockerfile` is Dockerfile to train model & compile forked TensorRT sources with support of `Hardshink` operator.

`data` contains .pgm MNIST samples needed for sample to run.

# Usage

1. Build docker image with `docker build -f .\Dockerfile --build-arg CUDA_VERSION=11.1 --tag=tensorrt-ubuntu-1804 .`. During build, model with custom layer is trained & forked TensorRT with implemented `Hardshrink` operator is built from sources.
2. Run container in interactive mode `docker run --rm -it --name hw4trt --env NVIDIA_DISABLE_REQUIRE=1 --gpus '"device=0"' --cap-add=SYS_ADMIN --ipc=host tensorrt-ubuntu-1804
3. Run `cd /wokrspace/TensorRT/build/out && ./sample_onnx_mnist_hardshrink --fp16 --datadir /workspace/TensorRT/data/`

# Issues

### 503 HTTP error for MNIST downloading

`TensorRT/samples/python/scripts/download_mnist_pgms.py` throws `urllib.error.HTTPError: HTTP Error 503: Service Unavailable` (despite that files could be downloaded with browser).
So, .pgm files are stored in `data` and copied into the image.

### INVALID_ARGUMENT

Unfortunately, the pipeline above leads to the following error during sample execution

    [04/14/2021-17:26:12] [E] [TRT] INVALID_ARGUMENT: Cannot find binding of given name: conv1
    Segmentation fault

The error persists on TensorRT/master with plain CNN model without custom Hardshink layer usage, so it seems that the torch.onnx.export usage in `examples/mnist/main.py` is incorrect.
