import pathlib
import glob
import numpy as np
import cv2


from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path: pathlib.Path = pathlib.Path(
    './models/mediapipe/pose-estimation/pose_landmarker_lite.task')
images_dir: pathlib.Path = pathlib.Path('assets/Mannequin/RGB')
# images_dir = pathlib.Path('assets/RSV/RGB')
images_save_dir: pathlib.Path = pathlib.Path(
    'assets/Mannequin/keypoints_output_images')
# images_save_dir = pathlib.Path('assets/RSV/keypoints_output_images')

MinConfidence = 0.7
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    min_pose_detection_confidence=MinConfidence,)

detector = PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def detect_landmarks(im_path: pathlib.Path, detector: vision.PoseLandmarker)->np.ndarray:
    # Load the input image from an image file.
    mp_image = mp.Image.create_from_file(f'{im_path.resolve()}')

    # Perform pose landmarking on the provided single image.
    # The pose landmarker must be created with the image mode.
    pose_landmarker_result = detector.detect(mp_image)
    # print(pose_landmarker_result)
    #Ensure that we send only the RGB channels if the loaded image is of RGBA format
    annotated_image = draw_landmarks_on_image(
            mp_image.numpy_view()[:,:,:3], pose_landmarker_result)
    
    return annotated_image


if __name__ == '__main__':
    types = ('*.png', '*.jpg')
    images_glob = []
    for type in types:
        glob_result = images_dir.glob(type)
        images_glob.extend(glob_result)

    images_save_dir.mkdir(parents=True, exist_ok=True)
   
    
    for im_path in images_glob:
        keypoints_im_path = images_save_dir.joinpath(
            f'{im_path.stem}-keypoints.jpg')
        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(f'{im_path.resolve()}')

        annotated_image = detect_landmarks(im_path, detector)

        # # Perform pose landmarking on the provided single image.
        # # The pose landmarker must be created with the image mode.
        # pose_landmarker_result = detector.detect(mp_image)

        # # STEP 5: Process the detection result. In this case, visualize it.
        # annotated_image = draw_landmarks_on_image(
        #     mp_image.numpy_view(), pose_landmarker_result)
        
        cv2.imwrite(f'{keypoints_im_path}', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
