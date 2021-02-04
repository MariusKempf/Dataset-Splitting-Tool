# Dataset-Splitting-Tool
Splitting datasets into n subsets

*Note: Until now only the datasets MNIST and Chest-XRAY are supported!*

This tool aims to split a dataset in multiple subsets.
My personal intention behind this tool is, to use the sliced datasets for experiments in the
field of Federated Learning where you have data distributed over multiple instances/machines.

```bash
# MNIST
python main.py -n 3 -d mnist -p /home/<user>/datasets
# XRAY
python main.py -n 3 -d xray -p /home/<user>/datasets
```

...

The scripts are indented to store the data in such way, that it can easily be used by the PyTorch
[ImageFolder](https://pytorch.org/vision/0.8/datasets.html#imagefolder).
The following tree gives an example.
```
The scripts produce the following structures:

data/
├──split_0
|   ├── train/
|   |   ├── class_0/
|   |   |   ├── 001.jpg
|   |   |   ├── 002.jpg
|   |   |   └── 003.jpg
|   |   └── class_m/
|   |       ├── 004.jpg
|   |       └── 005.jpg
|   └── test/
|       ├── class_0/
|       |   ├── 006.jpg
|       |   └── 007.jpg
|       └── class_m/
|           ├── 008.jpg
|           └── 009.jpg
...
|
├──split_n
|   ├── train/
|   |   ├── class_0/
...
```

#### Datasets

- [MNIST](http://yann.lecun.com/exdb/mnist/) by Yann LeCun
- [Chest-XRAY](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) by Kaggle

More dataset will be added in the future - feel free to add some yourself! :relaxed:
