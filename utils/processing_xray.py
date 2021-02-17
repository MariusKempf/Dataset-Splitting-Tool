import os
import zipfile
import numpy as np
from argparse import Namespace

from utils.utilities import build_dir_structure


def run_xray_process(args: Namespace,
                     data_types):
    # download
    if not _download_xray(args):
        _clean_up(args, full=True)
        return False

    # iterate all data types given by dataset
    for data_type in data_types:
        print(f'Chest-XRAY process (data: {data_type}) [START]')

        # distribute data into splits
        _distribute_images(args, data_type=data_type)

        print(f'Chest-XRAY process (data: {data_type}) [DONE]')

    # final clean up
    _clean_up(args)


def _download_xray(args: Namespace):
    data_archive = os.path.join(args.path, 'archive.zip')

    # check if data was downloaded
    if not os.path.exists(os.path.join(args.data_path, 'chest_xray')):
        if os.path.exists(data_archive):
            print('Unzipping data [START]')
            with zipfile.ZipFile(data_archive, 'r') as zf:
                zf.extractall(args.data_path)
            print('Unzipping data [DONE]')
        else:
            print("[INFO] Data still needs to be downloaded!\n"
                  "Unfortunately, it is not possible to automate the download in this case.\n"
                  "Go to: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia\n"
                  "Save archive into the intended directory - then run script again!")
            return False
    return True


def _distribute_images(args: Namespace,
                       data_type):
    print(f'Reordering data [START]')
    classes = ['NORMAL', 'PNEUMONIA']
    build_dir_structure(args, data_type, classes)

    for label in classes:
        print(f'Processing class "{label}" ...')
        path = os.path.join(args.data_path, 'chest_xray', data_type, label)

        img_list = os.listdir(path)
        splits = np.array_split(img_list, args.splits)

        for idx, split in enumerate(splits):
            for file in split:
                filepath = str(path + '/' + file)
                target_dir = os.path.join(args.data_path, f'split_{idx}', data_type, str(label))
                os.system(f'mv {filepath} {target_dir}')
    print(f'Reordering data [DONE]')


def _clean_up(args, full=False):
    if full:
        # Full clean-up
        os.system(f'rm -d -r {args.data_path}')
    else:
        path = os.path.join(args.path, 'archive.zip')
        os.system(f'rm -d -r {path}')
        path = os.path.join(args.data_path, 'chest_xray')
        os.system(f'rm -d -r {path}')
