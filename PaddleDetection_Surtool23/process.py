#encoding=utf-8
import os
#from PaddleDetection_Surtool23.tools.surtool_infer import infer, run

import SimpleITK
import numpy as np
import cv2
from pandas import DataFrame
from pathlib import Path
from scipy.ndimage import center_of_mass, label
from pathlib import Path
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    DataFrameValidator,
)
from typing import (Tuple)
from evalutils.exceptions import ValidationError
import random
import json
import ast


import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_mlu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.slim import build_slim_model


####
# Toggle the variable below to debug locally. The final container would need to have execute_in_docker=True
####
execute_in_docker = False


class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print('File found: ' + str(path))
        if ((str(path)[-3:])) == 'mp4':
            if not path.is_file():
                raise IOError(
                    f"Could not load {fname} using {self.__class__.__qualname__}."
                )
                #cap = cv2.VideoCapture(str(fname))
            #return [{"video": cap, "path": fname}]
            return [{"path": fname}]

# only path valid
    def hash_video(self, input_video):
        pass


class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError(
                "The videos are not unique, please submit a unique video for "
                "each case."
            )

class Surgtoolloc_det(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("../test/"),
            output_file=Path("/output/surgical-tools.json") if execute_in_docker else Path(
                            "../output/surgical-tools.json"),
            validators=dict(
                input_video=(
                    #UniqueVideoValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        
        ###                                                                                                     ###
        ###  TODO: adapt the following part for creating your model and loading weights
        ###                                                                                                     ###
        
        
            
        self.FLAGS = self.parse_args()
        self.cfg = load_config(self.FLAGS.config)
        merge_args(self.cfg, self.FLAGS)
        merge_config(self.FLAGS.opt)
    
        # disable npu in config by default
        if 'use_npu' not in self.cfg:
            self.cfg.use_npu = False
    
        # disable xpu in config by default
        if 'use_xpu' not in self.cfg:
            self.cfg.use_xpu = False
    
        if 'use_gpu' not in self.cfg:
            self.cfg.use_gpu = False
    
        # disable mlu in config by default
        if 'use_mlu' not in self.cfg:
            self.cfg.use_mlu = False
    
        if self.cfg.use_gpu:
            place = paddle.set_device('gpu')
        elif self.cfg.use_npu:
            place = paddle.set_device('npu')
        elif self.cfg.use_xpu:
            place = paddle.set_device('xpu')
        elif self.cfg.use_mlu:
            place = paddle.set_device('mlu')
        else:
            place = paddle.set_device('cpu')
    
        if self.FLAGS.slim_config:
            self.cfg = build_slim_model(self.cfg, self.FLAGS.slim_config, mode='test')
    
        check_config(self.cfg)
        check_gpu(self.cfg.use_gpu)
        check_npu(self.cfg.use_npu)
        check_xpu(self.cfg.use_xpu)
        check_mlu(self.cfg.use_mlu)
        check_version()
        
        self.trainer = Trainer(self.cfg, mode='test')

        # load weights
        self.trainer.load_weights(self.cfg.weights)
        
        self.tool_list = ["needle_driver",
                          "monopolar_curved_scissor",
                          "cadiere_forceps",
                          "suction_irrigator",
                          "bipolar_forceps",
                          "force_bipolar",
                          "grasping_retractor",
                          "vessel_sealer",
                          "prograsp_forceps",
                          "permanent_cautery_hook_spatula",
                          "tip_up_fenestrated_grasper",
                          "clip_applier",
                          "stapler",
                          "bipolar_dissector"]

    def get_test_images(self, infer_dir, infer_img):
        """
        Get image path list in TEST mode
        """
        assert infer_img is not None or infer_dir is not None, \
            "--infer_img or --infer_dir should be set"
        assert infer_img is None or os.path.isfile(infer_img), \
                "{} is not a file".format(infer_img)
        assert infer_dir is None or os.path.isdir(infer_dir), \
                "{} is not a directory".format(infer_dir)
    
        # infer_img has a higher priority
        if infer_img and os.path.isfile(infer_img):
            return [infer_img]
    
        images = set()
        infer_dir = os.path.abspath(infer_dir)
        assert os.path.isdir(infer_dir), \
            "infer_dir {} is not a directory".format(infer_dir)
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
        images = list(images)
    
        assert len(images) > 0, "no image found in {}".format(infer_dir)
        logger.info("Found {} inference images in total.".format(len(images)))
    
        return images
    
    def parse_args(self):
        parser = ArgsParser()
        parser.add_argument(
            "--infer_dir",
            type=str,
            default=None,
            help="Directory for images to perform inference on.")
        parser.add_argument(
            "--infer_img",
            type=str,
            default=None,
            help="Image path, has higher priority over --infer_dir")
        parser.add_argument(
            "--output_dir",
            type=str,
            default="visualize_output",
            help="Directory for storing the output visualization files.")
        parser.add_argument(
            "--draw_threshold",
            type=float,
            default=0.5,
            help="Threshold to reserve the result for visualization.")
        parser.add_argument(
            "--slim_config",
            default=None,
            type=str,
            help="Configuration file of slim method.")
        parser.add_argument(
            "--use_vdl",
            type=bool,
            default=False,
            help="Whether to record the data to VisualDL.")
        parser.add_argument(
            '--vdl_log_dir',
            type=str,
            default="vdl_log_dir/image",
            help='VisualDL logging directory for image.')
        parser.add_argument(
            "--save_results",
            type=bool,
            default=False,
            help="Whether to save inference results to output_dir.")
        parser.add_argument(
            "--slice_infer",
            action='store_true',
            help="Whether to slice the image and merge the inference results for small object detection."
        )
        parser.add_argument(
            '--slice_size',
            nargs='+',
            type=int,
            default=[640, 640],
            help="Height of the sliced image.")
        parser.add_argument(
            "--overlap_ratio",
            nargs='+',
            type=float,
            default=[0.25, 0.25],
            help="Overlap height ratio of the sliced image.")
        parser.add_argument(
            "--combine_method",
            type=str,
            default='nms',
            help="Combine method of the sliced images' detection results, choose in ['nms', 'nmm', 'concat']."
        )
        parser.add_argument(
            "--match_threshold",
            type=float,
            default=0.6,
            help="Combine method matching threshold.")
        parser.add_argument(
            "--match_metric",
            type=str,
            default='ios',
            help="Combine method matching metric, choose in ['iou', 'ios'].")
        parser.add_argument(
            "--visualize",
            type=ast.literal_eval,
            default=True,
            help="Whether to save visualize results to output_dir.")
        args = parser.parse_args()
        return args
    
    def process_case(self, *, idx, case):
        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py

        # Write resulting candidates to result.json for this case
        return dict(type="Multiple 2D bounding boxes", boxes=scored_candidates, version={"major": 1, "minor": 0})

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

    def generate_bbox(self, frame_id, temp_images_path):
        # bbox coordinates are the four corners of a box: [x, y, 0.5]
        # Starting with top left as first corner, then following the clockwise sequence
        # origin is defined as the top left corner of the video frame
        # get inference images
        images = self.get_test_images(None, os.path.join(temp_images_path, f"frame_{frame_id}.jpg"))
        _, bbox_list = self.trainer.predict(
            images,
            draw_threshold=self.FLAGS.draw_threshold,
            output_dir=self.FLAGS.output_dir,
            save_results=self.FLAGS.save_results,
            visualize=self.FLAGS.visualize)
            
        predictions = []
        for box in bbox_list:
            # fetch class name
            category_id = box['category_id']
            #print(category_id)
            name = f'slice_nr_{frame_id}_' + self.tool_list[category_id]

            # transform to corner format
            xmin, ymin, w, h = box['bbox']
            xmax, ymax = xmin + w, ymin + h
            
            # À©ÕÅ1.03±¶
            x_center = xmin + (w / 2)
            y_center = ymin + (h / 2)
            new_w = 1.03 * w
            new_h = 1.03 * h
            new_xmin = x_center - (new_w / 2)
            new_ymin = y_center - (new_h / 2)
            xmin, ymin, w, h = new_xmin, new_ymin, new_w, new_h
            xmax, ymax = xmin + w, ymin + h
            
            bbox = []
            bbox.append([xmin, ymin, 0.5])
            bbox.append([xmax, ymin, 0.5])
            bbox.append([xmax, ymax, 0.5])
            bbox.append([xmin, ymax, 0.5])
            # bbox = [[54.7, 95.5, 0.5],
            #         [92.6, 95.5, 0.5],
            #         [92.6, 136.1, 0.5],
            #         [54.7, 136.1, 0.5]]
            score = box['score']
            prediction = {"corners": bbox, "name": name, "probability": score}
            #print(prediction)
            predictions.append(prediction)
        return predictions

    def predict(self, fname) -> DataFrame:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
        print('Video file to be loaded: ' + str(fname))
        cap = cv2.VideoCapture(str(fname))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("num_frames{}".format(num_frames))

        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###

        # video2images
        temp_images_path = "./temp_images"
        if not os.path.exists(temp_images_path):
            os.mkdir(temp_images_path)
        
        
        if not cap.isOpened():
            print("unable to open video!")
            exit()
       
        
        all_frames_predicted_outputs = []
        for frame_number in range(num_frames):
            # set frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # next frame
            ret, frame = cap.read()

            # reach th end
            if not ret:
                continue

            # save image
            filename = os.path.join(temp_images_path, f"frame_{frame_number}.jpg")
            print("saved {}".format(filename))
            cv2.imwrite(filename, frame)
            
            tool_detections = self.generate_bbox(frame_number, temp_images_path)
            all_frames_predicted_outputs += tool_detections
            
            # delete temp images
            os.remove(os.path.join(temp_images_path, f"frame_{frame_number}.jpg"))
            
        # close video file
        cap.release()
        

        return all_frames_predicted_outputs


if __name__ == "__main__":
    Surgtoolloc_det().process()
    #infer()
