from argparse import Namespace

from utils.processing_mnist import run_mnist_process
from utils.processing_xray import run_xray_process


def trigger_pipeline(args: Namespace):
    """Starts dataset specific workflow"""

    if args.dataset == 'mnist':
        run_mnist_process(args, data_types=['train', 'test'])
    #elif args.dataset == 'chest-xray':
    elif args.dataset == 'xray':
        run_xray_process(args, data_types=['train', 'test', 'val'])
    else:
        print(f'[INFO] Dataset not supported yet! Stopping ...')
