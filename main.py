import os

from argparse import Namespace, ArgumentParser
from utils.trigger import trigger_pipeline

from utils.utilities import build_dir_structure

from utils.mnist_process import _make_splits


def main_cli():
    parser = ArgumentParser(
        description='''Tool to split MNIST data into multiple subsets - train and test data '''
    )
    parser.add_argument('-d', '--dataset', required=True,
                        default='mnist', type=str,
                        help='Choose dataset you want to process'
                        )
    parser.add_argument('-n', '--splits', required=False,
                        default=3, type=int,
                        help='Number of splits to generate'
                        )
    parser.add_argument('-p', '--path', required=False,
                        default='./', type=str,
                        help='Path where to save splits'
                        )
    arguments = parser.parse_args()
    arguments.data_path = os.path.join(arguments.path, arguments.dataset)
    return arguments


def main(args: Namespace):

    # create dataset and subset directories (if do not yet exist)
    if not os.path.exists(args.data_path)\
            and not os.path.exists(os.path.join(args.data_path, 'split_0')):
        os.mkdir(args.data_path)
        for i in range(args.splits):
            os.mkdir(os.path.join(args.data_path, f'split_{i}'))

    # Start processing pipelines
    trigger_pipeline(args, data_type='train')
    trigger_pipeline(args, data_type='test')

    #labels = [0,1,2,3,4]
    #build_dir_structure(args, 'train', labels)
    #build_dir_structure(args, labels)

    #_make_splits(args, 'train')

if __name__ == '__main__':
    args = main_cli()
    main(args)
