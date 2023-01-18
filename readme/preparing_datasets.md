# Preparing Datasets
## ImageNet and CIFAR datasets
* Download the original train and test set for [ImageNet](https://image-net.org/download.php) & [ImageNet-C](https://zenodo.org/record/2235448#.Yn5OTrozZhE) datasets.
* Download the original train and test set for [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) & [CIFAR-10C](https://zenodo.org/record/2535967#.Yn5QwbozZhE) datasets.

## KITTI dataset
* Download Clear (Original) [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).
* Download [KITTI-Fog/Rain](https://team.inria.fr/rits/computer-vision/weather-augment/) datasets.
* Super-impose snow on KITTI dataset through this [repository](https://github.com/hendrycks/robustness).
* Generate labels YOLO can use (see [Dataset directory structures](#dataset-directory-structures) subsection).

To generate labels YOLO can use from the original KITTI labels run

`python main.py --kitti_to_yolo_labels /path/to/original/kitti`

This is expecting the path to the original KITTI directory structure
```
path_to_specify
└── raw
    └── training
        ├── image_2
        └── label_2
```
Which will create a `yolo_style_labels` directory in the `raw` directory, containing
the KITTI labels in a format YOLO can use.

Structure the choosen dataset(s) as described [here](directory_scructures.md).