FROM paddlepaddle/paddle



RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip


COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt
RUN python -m pip install --user -r PaddleDetection_Surtool23/requirements.txt

COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm PaddleDetection_Surtool23 /opt/algorithm/PaddleDetection_Surtool23

ENTRYPOINT ["python", "-m", "process", "-c", "PaddleDetection_Surtool23/configs/semi_det/denseteacher/denseteacher_fcos_r50_fpn_coco_full.yml", "-o", "weights=PaddleDetection_Surtool23/model_weights/denseteacher_fcos_r50_fpn_coco_full/149.pdparams"]
