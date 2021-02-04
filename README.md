# MNIST-Splitting-Tool
Splitting a dataset into n subsets

**Note:** until now only MNIST is supported!

...

The scripts are indented to store the data in such way, that it can easily be used by PyTorch ImageFolder.
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

