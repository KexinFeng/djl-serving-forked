# Engine Configuration

This covers the available configurations for DJL and engines.

## DJL settings

DJLServing build on top of Deep Java Library (DJL). Here is a list of settings for DJL:

| Key                            | Type                | Description                                                                         |
|--------------------------------|---------------------|-------------------------------------------------------------------------------------|
| DJL_DEFAULT_ENGINE             | env var/system prop | The preferred engine for DJL if there are multiple engines, default: MXNet          |
| ai.djl.default_engine          | system prop         | The preferred engine for DJL if there are multiple engines, default: MXNet          |
| DJL_CACHE_DIR                  | env var/system prop | The cache directory for DJL: default: $HOME/.djl.ai/                                |
| ENGINE_CACHE_DIR               | env var/system prop | The cache directory for engine native libraries: default: $DJL_CACHE_DIR            |
| ai.djl.dataiterator.autoclose  | system prop         | Automatically close data set iterator, default: true                                |
| ai.djl.repository.zoo.location | system prop         | global model zoo search locations, not recommended                                  |
| offline                        | system prop         | Don't access network for downloading engine's native library and model zoo metadata |
| collect-memory                 | system prop         | Enable memory metric collection, default: false                                     |
| disableProgressBar             | system prop         | Disable progress bar, default: false                                                |

### PyTorch

| Key                                | Type                | Description                                                                                                                                                                          |
|------------------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PYTORCH_LIBRARY_PATH	              | env var/system prop | User provided custom PyTorch native library                                                                                                                                          |
| PYTORCH_VERSION                    | env var/system prop | PyTorch version to load                                                                                                                                                              |
| PYTORCH_EXTRA_LIBRARY_PATH         | env var/system prop | Custom pytorch library to load (e.g. torchneuron/torchvision/torchtext)                                                                                                              |
| PYTORCH_PRECXX11                   | env var/system prop | Load precxx11 libtorch                                                                                                                                                               |
| PYTORCH_FLAVOR                     | env var/system prop | To force override auto detection (e.g. cpu/cpu-precxx11/cu102/cu116-precxx11)                                                                                                        |
| PYTORCH_JIT_LOG_LEVEL              | env var	            | Enable [JIT logging](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/jit_log.h)                                                                                        |
| ai.djl.pytorch.native_helper	      | system prop         | A user provided custom loader class to help locate pytorch native resources                                                                                                          |
| ai.djl.pytorch.num_threads         | system prop         | Override OMP_NUM_THREAD environment variable                                                                                                                                         |
| ai.djl.pytorch.num_interop_threads | system prop         | Set PyTorch interop threads                                                                                                                                                          |
| ai.djl.pytorch.graph_optimizer     | system prop         | Enable/Disable JIT execution optimize, default: true. See: https://github.com/deepjavalibrary/djl/blob/master/docs/development/inference_performance_optimization.md#graph-optimizer |
| ai.djl.pytorch.cudnn_benchmark     | system prop         | To speed up ConvNN related model loading, default: false                                                                                                                             |
| ai.djl.pytorch.use_mkldnn          | system prop         | Enable MKLDNN, default: false, not recommended, use with your own risk                                                                                                               |

### TensorFlow

| Key                         | Type                | Description                                       |
|-----------------------------|---------------------|---------------------------------------------------|
| TENSORFLOW_LIBRARY_PATH     | env var/system prop | User provided custom TensorFlow native library    |
| TENSORRT_EXTRA_LIBRARY_PATH | env var/system prop | Extra TensorFlow custom operators library to load |
| TF_CPP_MIN_LOG_LEVEL        | env var             | TensorFlow log level                              |
| ai.djl.tensorflow.debug     | env var             | Enable devicePlacement logging, default: false    |

### MXNet

| Key                               | Type                | Description                                                                    |
|-----------------------------------|---------------------|--------------------------------------------------------------------------------|
| MXNET_LIBRARY_PATH                | env var/system prop | User provided custom MXNet native library                                      |
| MXNET_VERSION                     | env var/system prop | The version of custom MXNet build                                              |
| MXNET_EXTRA_LIBRARY_PATH          | env var/system prop | Load extra MXNet custom libraries, e.g. Elastice Inference                     |
| MXNET_EXTRA_LIBRARY_VERBOSE       | env var/system prop | Set verbosity for MXNet custom library                                         |
| ai.djl.mxnet.static_alloc         | system prop         | CachedOp options, default: true                                                |
| ai.djl.mxnet.static_shape         | system prop         | CachedOp options, default: true                                                |
| ai.djl.use_local_parameter_server | system prop         | Use java parameter server instead of MXNet native implemention, default: false |

### PaddlePaddle

| Key                                     | Type                | Description                                      |
|-----------------------------------------|---------------------|--------------------------------------------------|
| PADDLE_LIBRARY_PATH                     | env var/system prop | User provided custom PaddlePaddle native library |
| ai.djl.paddlepaddle.disable_alternative | system prop         | Disable alternative engine                       |

### Huggingface tokenizers

| Key              | Type    | Description                                               |
|------------------|---------|-----------------------------------------------------------|
| TOKENIZERS_CACHE | env var | User provided custom Huggingface tokenizer native library |

### Python

| Key                               | Type                | Description                                                                                                                                            |
|-----------------------------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| PYTHON_EXECUTABLE                 | env var             | The location is python executable, default: python                                                                                                     |
| DJL_ENTRY_POINT                   | env var             | The entrypoint python file or module, default: model.py                                                                                                |
| MODEL_LOADING_TIMEOUT             | env var             | Python worker load model timeout: default: 240 seconds                                                                                                 |
| PREDICT_TIMEOUT                   | env var             | Python predict call timeout, default: 120 seconds                                                                                                      |
| DJL_VENV_DIR                      | env var/system prop | The venv directory, default: $DJL_CACHE_DIR/venv                                                                                                       |
| ai.djl.python.disable_alternative | system prop         | Disable alternative engine                                                                                                                             |
| TENSOR_PARALLEL_DEGREE            | env var             | Set tensor parallel degree.<br>For mpi mode, the default is number of accelerators.<br>Use "max" for non-mpi mode to use all GPUs for tensor parallel. |

## Engine specific settings

DJL support 12 deep learning frameworks, each framework has their own settings. Please refer to
each framework’s document for detail.

A common setting for most of the engines is ``OMP_NUM_THREADS``, for the best throughput,
DJLServing set this to 1 by default. For some engines (e.g. **MXNet**, this value must be one).
Since this is a global environment variable, setting this value will impact all other engines.

The follow table show some engine specific environment variables that is override by default by DJLServing:

| Key                    | Engine     | Description                                         |
|------------------------|------------|-----------------------------------------------------|
| TF_NUM_INTEROP_THREADS | TensorFlow | default 1, OMP_NUM_THREADS will override this value |
| TF_NUM_INTRAOP_THREADS | TensorFlow | default 1                                           |
| TF_CPP_MIN_LOG_LEVEL	  | TensorFlow | default 1                                           |
| MXNET_ENGINE_TYPE      | MXNet      | this value must be `NaiveEngine`                    |

