#Example python /home/ashok/Workspace/LearnMLNN/projects/Infant-pose-estimation/pipeline/batch_volume_recorder.py -bl blender312 -rdir ./assets/Ashok-RGB2D/RGB/ -ddir ./assets/Ashok-RGB2D/DEPTH/ -sdir ./assets/Ashok-RGB2D/SEGMENTATION/ -kpd ./output/Ashok-RGB2D/mediapipe_out/
import sys
import pathlib
import subprocess

import argparse

BLENDER_TEMPLATE_PATH: pathlib.Path = pathlib.Path('/home/ashok/Workspace/LearnMLNN/projects/Infant-pose-estimation/pipeline/TEMPLATE-ALIGNMENT/blends/Keypoints-Fitting-MediaPipe-LSE.blend')
PYTHON_SCRIPT_PATH: pathlib.Path = pathlib.Path('/home/ashok/Workspace/LearnMLNN/projects/Infant-pose-estimation/pipeline/TEMPLATE-ALIGNMENT/bl-scripts/Keypoints-Fitting-LSE.py')
CAMERA_INTRINSICS_PATH: pathlib.Path = pathlib.Path('/home/ashok/Workspace/LearnMLNN/projects/Infant-pose-estimation/pipeline/DEPTH-TO-PLY/azure_depth_camera_intrinsics.npy')

def getArguments()  -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-bl', '--blender', dest='blender', type=pathlib.Path, required=True, help="Blender executable path or global binary name")
    parser.add_argument('-rdir', '--rgb-dir', dest='rgbdir', type=pathlib.Path, required=True, help="Directory path where the rgb images can be found")
    parser.add_argument('-ddir', '--depth-dir', dest='depthdir', type=pathlib.Path, required=True, help = "Directory path where the depth images are found")
    parser.add_argument('-sdir', '--segmentation-dir', dest='segmentationdir', type=pathlib.Path, required=True, help = "Directory path where the segmentation images are found")
    parser.add_argument('-kpd', '--kps-dir', dest='keypointsdir', type=pathlib.Path, required=True, help = "Directory path where the keypoint json files can be found")
    args = parser.parse_known_args(sys.argv)[0]
    return args

def volumeExtractor(blender_path: pathlib.Path, rgb_path:pathlib.Path, depth_path: pathlib.Path, segmentation_path: pathlib.Path, keypoints_path: pathlib.Path)->None:
    blend_file: pathlib.Path = keypoint_path.parent.joinpath('pipeline_results').joinpath(f'{keypoints_path.stem}.blend')
    volume_file: pathlib.Path = keypoint_path.parent.joinpath('pipeline_results').joinpath(f'{keypoints_path.stem}.log')
    blender_command: list[str] = []
    
    blender_command.append(f'{blender_path}')
    blender_command.append(f'{BLENDER_TEMPLATE_PATH}')
    
    blender_command.append('-b')
    
    blender_command.append('--python')
    blender_command.append(f'{PYTHON_SCRIPT_PATH}')
    blender_command.append('--')

    blender_command.append('-ci')
    blender_command.append(f'{CAMERA_INTRINSICS_PATH}')


    blender_command.append('-rimg')
    blender_command.append(f'{rgb_path}')

    blender_command.append('-dimg')
    blender_command.append(f'{depth_path}')

    blender_command.append('-simg')
    blender_command.append(f'{segmentation_path}')

    blender_command.append('-kpjs')
    blender_command.append(f'{keypoints_path}')
    blender_command_str = ' '.join(['%s'%(v) for v in blender_command])

    if(not blend_file.exists()):
        sproc = subprocess.run(blender_command_str, shell=True, text=True, check=True)

    if(volume_file.exists()):
        f = open(f'{volume_file}')
        try:
            volume_line: list[str] = f.readlines()[6].split(':')#The line with the volume data
            volume_value: float = float(volume_line[1].strip())
        except IndexError:
            volume_value: float = 0
        f.close()
        return volume_value
    else:
        return 0


if __name__ == '__main__':
    args:argparse.Namespace = getArguments()
    volume_values_file_path: pathlib.Path = args.rgbdir.parent.joinpath('volumedata.csv')
    volume_file = open(f'{volume_values_file_path}', 'w')
    rgb_files: list[pathlib.Path] = sorted(list(args.rgbdir.glob('*.png')))
    volume_file.write('Frame Index, Volume(m3);\n')
    for rgb_im_path in rgb_files:
        filename: str = rgb_im_path.stem
        file_name_index: str = filename.split('-')[0]
        depth_path: pathlib.Path = args.depthdir.joinpath(f'{file_name_index}-DEPTH.png')
        segmentation_path: pathlib.Path = args.segmentationdir.joinpath(f'{file_name_index}-SEGMENTATION.png')
        keypoint_path: pathlib.Path = args.keypointsdir.joinpath(f'{filename}_keypoints.json')
        volume_recorded: float = volumeExtractor(args.blender, rgb_im_path, depth_path, segmentation_path, keypoint_path)
        volume_file.write(f'{file_name_index}, {volume_recorded};\n')
        volume_file.flush()
        # break
    
    volume_file.close()

# blender312 
# ./TEMPLATE-ALIGNMENT/blends/Keypoints-Fitting-MediaPipe-LSE.blend 
# -b --python ./TEMPLATE-ALIGNMENT/bl-scripts/Keypoints-Fitting-LSE.py 
# -- 
# -dimg ./assets/Synthetic/DEPTH/1-DEPTH.png 
# -kpjs output/Synthetic/mediapipe_out/1-RGB_keypoints.json  
# -rimg ./assets/Synthetic/RGB/1-RGB.png 
# -simg ./assets/Synthetic/SEGMENTATION/1-SEGMENTATION.png

RGB_FOLDER:pathlib.Path = pathlib.Path(__file__)
