import pathlib

def get_images(images_dir: pathlib.Path)->list:
    types = ('*.png', '*.jpg')
    images_glob = []
    for type in types:
        glob_result = images_dir.glob(type)
        images_glob.extend(glob_result)

    return images_glob