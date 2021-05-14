import os
import gzip
import struct
import pathlib
import requests
import numpy as np
import pandas as pd

from argparse import Namespace
from PIL import Image

from utils.utilities import build_dir_structure


def run_mnist_process(args: Namespace):

    for data_type in ['train, test']:
        print(f'Preparing MNIST (data: {data_type}) [START]')

        # download
        _download_mnist(args, data_type=data_type)

        # load images
        print('Load images [START]')
        images, labels = _load_mnist(args, data_type=data_type)
        print('Load images [DONE]')

        # generate images and label list
        print('preparing images and labels [START]')
        _generate_images(args, data_type, images, labels)
        _generate_labellist(args, data_type, labels)
        print('preparing images and labels [DONE]')

        print(f'Preparing MNIST (data: {data_type}) [DONE]')

    # split training data
    _make_training_splits(args)

    # split validation data & seperate test data
    print(f'MNIST splitting validataion data [START]')

    print(f'MNIST splitting validation data [DONE]')

    # final clean up
    #    _clean_up(args)


def _download_mnist(args: Namespace,
                    data_type: str):
    print(f'Data download [START]')
    raw_dir = os.path.join(args.data_path, 'raw')
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    if data_type == 'test':
        data_type = 't10k'

    baseurl = 'http://yann.lecun.com/exdb/mnist'
    urls = [
        f'{baseurl}/{data_type}-images-idx3-ubyte.gz',
        f'{baseurl}/{data_type}-labels-idx1-ubyte.gz'
    ]
    # run downloads
    for url in urls:
        print(f'Downloading from: {url}')
        filepath = os.path.join(raw_dir, pathlib.Path(url).name)
        if not os.path.exists(filepath):
            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
            else:
                print(f'Response status_code: {response.status_code}')
                raise AssertionError ('MNIST download failed!')
    print(f'Data download [DONE]')


def _load_mnist(args: Namespace,
                data_type: str):

    if data_type == 'test':
        data_type = 't10k'

    raw_dir = os.path.join(args.data_path, 'raw')
    paths = [
        os.path.join(raw_dir, f'{data_type}-images-idx3-ubyte.gz'),
        os.path.join(raw_dir, f'{data_type}-labels-idx1-ubyte.gz')
    ]
    return _load(paths)


def _load(paths):
    x_path, y_path = paths
    with gzip.open(x_path) as fx, gzip.open(y_path) as fy:
        fx.read(4)
        fy.read(4)
        N, = struct.unpack('>i', fy.read(4))
        if N != struct.unpack('>i', fx.read(4))[0]:
            raise RuntimeError('wrong pair of MNIST images and labels')
        fx.read(8)

        images = np.empty((N, 784), dtype=np.uint8)
        labels = np.empty(N, dtype=np.uint8)

        for i in range(N):
            labels[i] = ord(fy.read(1))
            for j in range(784):
                images[i, j] = ord(fx.read(1))
    return images, labels


def _generate_images(args: Namespace,
                     data_type: str,
                     images: np,
                     labels: np):

    path = os.path.join(args.data_path, 'processed', 'images', data_type)
    os.makedirs(path)
    for (i, image), label in zip(enumerate(images), labels):
        filepath = os.path.join(path, f'{label}_{i}.jpg')
        Image.fromarray(image.reshape(28, 28)).save(filepath)


def _generate_labellist(args: Namespace,
                        data_type: str,
                        labels: np):

    path = os.path.join(args.data_path, 'processed', 'labels', data_type)
    os.makedirs(path)
    img_paths = [
        f'{label}_{i}.jpg' for i, label in enumerate(labels)
    ]
    df = pd.DataFrame({'name': img_paths, 'target': labels.tolist()})
    df.to_csv(path + f'/{data_type}.csv', index=False, header=False)


def _make_training_splits(args: Namespace,):
    print(f'Splitting training data [START]')

    path_labels = os.path.join(args.data_path, 'processed', 'labels', 'train')
    df = pd.read_csv(path_labels + '/train.csv',
                     names=['file', 'label'], header=None)

    classes = sorted(df['label'].unique())
    build_dir_structure(args, 'train', classes)

    path_imgs = os.path.join(args.data_path, 'processed', 'images', 'train')
    for label in classes:
        print(f'Processing class "{label}" ...')
        tmp = df.loc[df['label'] == label]
        splits = np.array_split(tmp.file.to_list(), args.splits)

        for idx, split in enumerate(splits):
            for file in split:
                filepath = str(path_imgs + '/' + file)
                target_dir = os.path.join(args.data_path, f'split_{idx}', 'train', str(label))
                os.system(f'mv {filepath} {target_dir}')

    print(f'Splitting training data [DONE]')


def _make_validation_splits(args: Namespace):
    print(f'MNIST splitting validataion data [START]')
    print('MNIST comes with 10k images for testing/validation - these can be split for test/val')
    
    test_fraction = float(input("What fraction (%) of the 10.000 images do you want for final testing (i.e. 0.25) ? "))

    df_validation = _seperate_test_data(args, test_fraction)

    print('#'*50, df_validation.shape)

    classes = sorted(df_validation['label'].unique())
    build_dir_structure(args, 'val', classes)

    path_imgs = os.path.join(args.data_path, 'processed', 'images', 'val')
    for label in classes:
        print(f'Processing class "{label}" ...')
        tmp = df_validation.loc[df_validation['label'] == label]
        splits = np.array_split(tmp.file.to_list(), args.splits)

        for idx, split in enumerate(splits):
            for file in split:
                filepath = str(path_imgs + '/' + file)
                target_dir = os.path.join(args.data_path, f'split_{idx}', 'val', str(label))
                os.system(f'mv {filepath} {target_dir}')

    print(f'MNIST splitting validation data [DONE]')


def _seperate_test_data(args : Namespace, test_fraction : float):
    """ """

    path_imgs = os.path.join(args.data_path, 'processed', 'images', 'test')
    path_labels = os.path.join(args.data_path, 'processed', 'labels', 'test')

    df = pd.read_csv(path_labels + '/test.csv', names=['file', 'label'], header=None)
    df_test = df.sample(frac=test_fraction)

    classes = sorted(df['label'].unique())

    print('#' * 50, df_test.shape)
    
    # create folder structure for test data
    for c in classes:
        os.makedirs(os.path.join('test', str(c)))

    # process imgs & labels
    for label in classes:
        tmp = df_test.loc[df_test['label'] == label]
        file_list  = tmp.file.to_list()

        for file in file_list:
            filepath = path_imgs + '/' + file
            target_dir = os.path.join(args.data_path, 'test', str(label))
            os.system(f'mv {filepath} {target_dir}')

    # return dataframe without the entries used for testing data
    return df.drop(df_test.index)


def _clean_up(args: Namespace):
    for folder in ['raw', 'processed']:
        path = os.path.join(args.data_path, folder)
        os.system(f'rm -d -r {path}')
