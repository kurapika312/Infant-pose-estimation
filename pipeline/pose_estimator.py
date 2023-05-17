import pathlib 
import numpy as np
import cv2

from utils import get_images
from pediatrics_keypoints_task import predictor, detect_landmarks as detect_landmarks_detectron2
from mediapipe_pose_estimation import detector, detect_landmarks as detect_landmarks_mediapipe




if __name__ == '__main__':
    input_images_dir = pathlib.Path('assets/Mannequin/RGB')
    # input_images_dir = pathlib.Path('assets/RSV/color')

    output_images_dir = pathlib.Path('./output').joinpath(input_images_dir.parent.stem)
    images_glob = get_images(input_images_dir)

    for im_path in images_glob:
        annotated_detectron2: np.ndarray = detect_landmarks_detectron2(im_path, predictor)
        annotated_mediapipe: np.ndarray = detect_landmarks_mediapipe(im_path, detector)

        save_annotation_detectron2: pathlib.Path = output_images_dir.joinpath('detectron2_out').joinpath(f'{im_path.stem}-keypoints.jpg')
        save_annotation_mediapipe: pathlib.Path = output_images_dir.joinpath('mediapipe_out').joinpath(f'{im_path.stem}-keypoints.jpg')

        save_annotation_detectron2.parent.mkdir(parents=True, exist_ok=True)
        save_annotation_mediapipe.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(f'{save_annotation_detectron2}', annotated_detectron2)
        cv2.imwrite(f'{save_annotation_mediapipe}', annotated_mediapipe)










