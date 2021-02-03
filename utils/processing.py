from argparse import Namespace
from utils.downloads import download


def pipeline(args: Namespace):

    # download
    download(args)

    pass

