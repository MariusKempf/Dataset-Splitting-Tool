import os
import pathlib
import requests

from argparse import Namespace


def download(args: Namespace):

    if args.dataset == 'mnist':
        print('[START] MNIST download')
        _download_mnist(args)
    else:
        print('[FAILED]')

    pass


def _download_mnist(args):

    raw_dir = os.path.join(args.data_path, 'raw')
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    baseurl = 'http://yann.lecun.com/exdb/mnist'
    for data_type in ['train', 't10k']:
        urls = [
            '{}/{}-images-idx3-ubyte.gz'.format(baseurl, data_type),
            '{}/{}-labels-idx1-ubyte.gz'.format(baseurl, data_type)
        ]
        # run downloads
        for url in urls:
            filepath = os.path.join(raw_dir, pathlib.Path(url).name)
            if not os.path.exists(filepath):
                response = requests.get(url)
                if response.status_code == 200:
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
