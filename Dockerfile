FROM paddlecloud/paddledetection:2.4-gpu-cuda11.2-cudnn8-latest

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

ENV HOME=/opt/algorithm


RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user pip
RUN python3 -m pip install --user pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
COPY --chown=algorithm:algorithm PaddleDetection_Surtool23 /opt/algorithm/PaddleDetection_Surtool23
RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ -r PaddleDetection_Surtool23/requirements.txt
RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

WORKDIR /opt/algorithm/PaddleDetection_Surtool23

#COPY --chown=algorithm:algorithm process.py /opt/algorithm/

# 设置默认命令
ENTRYPOINT ["python3", "-m", "process", "-c", "configs/semi_det/denseteacher/denseteacher_ppyoloe_plus_crn_x_coco_full.yml", "-o", "weights=model_weights/denseteacher_ppyoloe_plus_crn_x_coco_full/best_model.pdparams"]
#ENTRYPOINT python -m process $0 $@
