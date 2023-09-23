# Surgical Tool Localization in Endoscopic Videos: A Novel Approach Using Dense Teacher Models

## Install 

Our code is based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).To run our code, you must first install PaddlePaddle and PaddleDetection.

### Requirements:

- PaddlePaddle 2.2
- OS 64 bit
- Python 3(3.5.1+/3.6/3.7/3.8/3.9/3.10)，64 bit
- pip/pip3(9.0.1+), 64 bit
- CUDA >= 10.2
- cuDNN >= 7.6


Dependency of PaddleDetection and PaddlePaddle:

| PaddleDetection version | PaddlePaddle version |                     tips                     |
| :---------------------: | :------------------: | :------------------------------------------: |
|         develop         |       >= 2.3.2       |        Dygraph mode is set as default        |
|       release/2.6       |       >= 2.3.2       |        Dygraph mode is set as default        |
|       release/2.5       |       >= 2.2.2       |        Dygraph mode is set as default        |
|       release/2.4       |       >= 2.2.2       |        Dygraph mode is set as default        |
|       release/2.3       |      >= 2.2.0rc      |        Dygraph mode is set as default        |
|       release/2.2       |       >= 2.1.2       |        Dygraph mode is set as default        |
|       release/2.1       |       >= 2.1.0       |        Dygraph mode is set as default        |
|       release/2.0       |       >= 2.0.1       |        Dygraph mode is set as default        |
|     release/2.0-rc      |       >= 2.0.1       |                      --                      |
|       release/0.5       |       >= 1.8.4       | Cascade R-CNN and SOLOv2 depends on 2.0.0.rc |
|       release/0.4       |       >= 1.8.4       |           PP-YOLO depends on 1.8.4           |
|       release/0.3       |        >=1.7         |                      --                      |

### Instruction

#### 1. Install PaddlePaddle

```
# CUDA10.2
python -m pip install paddlepaddle-gpu==2.3.2 -i https://mirror.baidu.com/pypi/simple

# CPU
python -m pip install paddlepaddle==2.3.2 -i https://mirror.baidu.com/pypi/simple
```

- For more CUDA version or environment to quick install, please refer to the [PaddlePaddle Quick Installation document](https://www.paddlepaddle.org.cn/install/quick)
- For more installation methods such as conda or compile with source code, please refer to the [installation document](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)

Please make sure that your PaddlePaddle is installed successfully and the version is not lower than the required version. Use the following command to verify.

```
# check
>>> import paddle
>>> paddle.utils.run_check()

# confirm the paddle's version
python -c "import paddle; print(paddle.__version__)"
```

**Note**

1.  If you want to use PaddleDetection on multi-GPU, please install NCCL at first.

#### 2. Install PaddleDetection

**Note:** Installing via pip only supports Python3

```
cd PaddleDetection_Surtool23

# Install other dependencies
pip install -r requirements.txt

# Compile and install paddledet
python setup.py install

```

**Note**

1. If you are working on Windows OS, `pycocotools` installing may failed because of the origin version of cocoapi does not support windows, another version can be used used which only supports Python3:

   ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```

2. If you are using Python <= 3.6, `pycocotools` installing may failed with error like `distutils.errors.DistutilsError: Could not find suitable distribution for Requirement.parse('cython>=0.27.3')`, please install `cython` firstly, for example `pip install cython`

After installation, make sure the tests pass:

```shell
python ppdet/modeling/tests/test_architectures.py
```

If the tests are passed, the following information will be prompted:

```
.......
----------------------------------------------------------------------
Ran 7 tests in 12.816s
OK
```



### Install other requirements

Other requirements are:

- evalutils==0.3.1
- scikit-learn==0.24.2
- scipy

you could simply run:

```
cd ..
pip install -r requirements.txt
```



## Run

After completing the installation, you can run our code.

You can download our model using this [link](https://pan.baidu.com/s/12s_AFa78YZRqXTXz4rdIzg?pwd=23d5) and place it in the following directory: `PaddleDetection_Surtool23/model_weights/denseteacher_ppyoloe_plus_crn_x_coco_full/`

If the link is not working, please contact 1120201985@bit.edu.cn.

After downloading weight, you could run our code by:
```
cd PaddleDetection_Surtool23
python process.py -c configs/semi_det/denseteacher/denseteacher_ppyoloe_plus_crn_x_coco_full.yml -o weights=model_weights/denseteacher_ppyoloe_plus_crn_x_coco_full/best_model.pdparams
```

The results will be saved in `output/surgical-tools.json`.



## Build docker

When building the Docker image, please modify line 36 of `PaddleDetection_Surtool23/process.py`, change `False` to `True`

```
####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
####
# When building docker image, modify 'False' to 'True'
# execute_in_docker = False
execute_in_docker = True


class VideoLoader():
    def load(self, *, fname):
```

Then run the scripts:

```
# build the docker image
./build.sh

# test the docker image
./test.sh

# export the docker image
./export.sh
```

