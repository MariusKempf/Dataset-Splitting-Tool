from argparse import Namespace

from utils.processing_mnist import run_mnist_process
from utils.processing_xray import run_xray_process


def trigger_pipeline(args: Namespace):
    """Starts dataset specific workflow"""

    if args.dataset == 'mnist':
        print(f'[INFO] DATASET SPLITTING TOOL - processing: {args.dataset}')
        run_mnist_process(args)
    elif args.dataset == 'chest-xray':
        print(f'[INFO] DATASET SPLITTING TOOL - processing: {args.dataset}')
        run_xray_process(args)
    else:
        print(f'[INFO] Dataset not supported yet! Stopping ...')
