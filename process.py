from PaddleDetection_Surtool23.tools.surtool_infer import infer

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
import os


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
            input_path=Path("/input/") if execute_in_docker else Path("./test/"),
            output_file=Path("/output/surgical-tools.json") if execute_in_docker else Path(
                            "./output/surgical-tools.json"),
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
        bbox_list = infer(os.path.join(temp_images_path, f"frame_{frame_id}.jpg"))
        predictions = []
        for box in bbox_list:
            # fetch class name
            category_id = box['category_id']
            name = f'slice_nr_{frame_id}_' + self.tool_list[category_id]

            # transform to corner format
            xmin, ymin, w, h = box['bbox']
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
            print(prediction)
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

        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###

        # video2images
        temp_images_path = "./temp_images"
        if not os.path.exists(temp_images_path):
            os.mkdir(temp_images_path)
        
        # delete former images
        for filename in os.listdir(temp_images_path):
            file_path = os.path.join(temp_images_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"unable to delete {file_path}: {e}")
        
        if not cap.isOpened():
            print("unable to open video!")
            exit()

        # save each frame as an image file
        frame_count = 0
        while True:
            # next frame
            ret, frame = cap.read()

            # reach th end
            if not ret:
                break

            # save image
            filename = os.path.join(temp_images_path, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)

            # update counter
            frame_count += 1

        # close video file
        cap.release()
       
        
        all_frames_predicted_outputs = []
        for fid in range(frame_count):
            tool_detections = self.generate_bbox(fid, temp_images_path)
            all_frames_predicted_outputs += tool_detections

        return all_frames_predicted_outputs


if __name__ == "__main__":
    Surgtoolloc_det().process()
    #infer()
