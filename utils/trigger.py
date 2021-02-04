from argparse import Namespace
from utils.mnist_process import run_mnist_process


def trigger_pipeline(args: Namespace,
                     data_type: str = 'train'):

    if args.dataset == 'mnist':
        run_mnist_process(args, data_type)
    else:
        print()
