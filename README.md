# DUA: Dynamic Unsupervised Adaptation (CVPR 2022)

This is the official repository for our paper: [The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization](https://openaccess.thecvf.com/content/CVPR2022/papers/Mirza_The_Norm_Must_Go_On_Dynamic_Unsupervised_Domain_Adaptation_by_CVPR_2022_paper.pdf)

DUA is an extremely simple method which only adapts the (1st and 2nd order) statistics of the Batch Normalization layer 
in an online manner to adapt to the out-of-distribution test data at test-time. Adapting only the statistics for 
Unsupervised Domain Adaptation makes DUA extremely fast and computation efficient. Moreover, 
DUA requires less than 1% of data from the target domain and no back propagation to achieve 
competitive (and often state-of-the-art) results when compared to strong baselines.

Short explanatory video about DUA is hosted [here](https://www.youtube.com/watch?v=fTe0Aqs-t7E).

# Installation

1) `git clone` this repository.
2) `pip install -r requirements.txt` to install required packages

# Running Experiments

[comment]: <> (We recommend first setting up user specific paths in the `PATHS` dictionary in `config.py`,)

[comment]: <> (by following the existing entry as an example and use `--usr` argument to set paths automatically.)

[comment]: <> (However, all experiments can also be run through explicit command)

[comment]: <> (line arguments. )
Before starting with running the experiments, please prepare the datasets through the instructions listed
[here](readme/preparing_datasets.md).

We provide code for reproducing CIFAR-10C / ImageNet-C / KITTI. These experiments 
can be run through the following example commands.  

### CIFAR-10C (WRN-40-2)
For running this experiment first download the [AugMix](https://arxiv.org/abs/1912.02781) pre-trained
[WRN-40-2 Checkpoint](https://drive.google.com/file/d/1wy7gSRsUZzCzj8QhmTbcnwmES_2kkNph/view).
```
python main.py --dataset cifar10 --ckpt_path path/to/checkpoint.pt --data_root root/path/for/cifar-10C
```
#### WRN - Results Cifar10C (Level-5 Severity)
| | data samples used| mean error | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------------- | ---:|---: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| source        |10000 |18.3|28.8| 22.9|26.2|9.5| 20.6|10.6|9.3|14.2|15.3|17.5|7.6|20.9|14.7|41.3|14.7|
| tent           |10000 |12.3|15.8|13.5|18.7|8.1|18.7|9.1|8.0|10.3|10.8|11.7|6.7|11.6|14.1|11.7|15.2|
| dua          |80|12.1|15.4|13.4|17.3|8.0|18.0|9.1|7.7|10.8|10.8|12.1|6.6|10.9|13.6|13.0|14.3|

### ImageNet-C (ResNet-18)
```
python main.py --dataset imagenet --data_root root/path/for/imagenet-c
```

### KITTI (YOLOv3)
```
python main.py --dataset kitti --data_root root/path/for/kitti
```
This will first train the network on the original KITTI dataset and then adapt separately to `Fog` and `Rain`. 
The current hyper-parameters are set to the default values used in the DUA paper, to experiment with other 
settings please refer to `main.py`.

#### To cite us: 
```bibtex
@InProceedings{mirza2022dua,
    author    = {Mirza, M. Jehanzeb and Micorek, Jakub and Possegger, Horst and Bischof, Horst},
    title     = {The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022}
}
```

Also read [DISC](https://openaccess.thecvf.com/content/CVPR2022W/V4AS/papers/Mirza_An_Efficient_Domain-Incremental_Learning_Approach_To_Drive_in_All_Weather_CVPRW_2022_paper.pdf), an extension of DUA - accepted at CVPR workshops. 
```bibtex
@InProceedings{mirza2022disc,
    author    = {Mirza, M. Jehanzeb and Masana, Marc and Possegger, Horst and Bischof, Horst},
    title     = {An Efficient Domain-Incremental Learning Approach To Drive in All Weather Conditions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year      = {2022}
}
```
