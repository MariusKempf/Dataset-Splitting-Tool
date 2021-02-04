import os
from argparse import Namespace


def build_dir_structure(args: Namespace,
                        data_type: str,
                         classes):

    for split in range(args.splits):
        for label in classes:
            path = os.path.join(args.data_path, f'split_{split}', data_type, str(label))
            if not os.path.exists(path):
                os.makedirs(path)
