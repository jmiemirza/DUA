
# DUA: Dynamic Unsupervised Adaptation (CVPR 2022)

This is the official repository for our paper: [The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization](https://openaccess.thecvf.com/content/CVPR2022/papers/Mirza_The_Norm_Must_Go_On_Dynamic_Unsupervised_Domain_Adaptation_by_CVPR_2022_paper.pdf)

DUA is an extremely simple method which only adapts the (1st and 2nd order) statistics of the Batch Normalization layer 
in an online manner to adapt to the out-of-distribution test data at test-time. Adapting only the statistics for 
Unsupervised Domain Adaptation makes DUA extremely fast and computation efficient. Moreover, 
DUA requires less than 1% of data from the target domain and no back propagation to achieve 
competitive (and often state-of-the-art) results when compared to strong baselines.

Short explanatory video about DUA is hosted [here](https://www.youtube.com/watch?v=fTe0Aqs-t7E).

Currently we provide example code for Cifar-10C experiments from our paper.


## Dataset

Download the [Cifar-10C dataset](https://zenodo.org/record/2535967).

## WRN-40-2
Part of the results in the paper are obtained by using [AugMix](https://arxiv.org/abs/1912.02781) WRN-40-2 from the [Robust Bench](https://robustbench.github.io/).
For running this experiment first download the [WRN-40-2 Checkpoint](https://drive.google.com/file/d/1wy7gSRsUZzCzj8QhmTbcnwmES_2kkNph/view).


### WRN - Results Cifar10C (Level-5 Severity)
|                                                            | data samples used| mean error | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------------- | ---:|---: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| source        |10000 |18.3|28.8| 22.9|26.2|9.5| 20.6|10.6|9.3|14.2|15.3|17.5|7.6|20.9|14.7|41.3|14.7|
| tent           |10000 |12.3|15.8|13.5|18.7|8.1|18.7|9.1|8.0|10.3|10.8|11.7|6.7|11.6|14.1|11.7|15.2|
| dua          |80|12.1|15.4|13.4|17.3|8.0|18.0|9.1|7.7|10.8|10.8|12.1|6.6|10.9|13.6|13.0|14.3|

**Usage**:
```python
python dua.py --model wrn --dataroot ROOT_PATH_FOR_CIFAR_10C_DATASET --ckpt_path PATH_FOR_DOWNLOADED_CHECKPOINT
```

## ResNet-26
We also use ResNet-26 for comparison with some baselines. To run this experiment, download the [ResNet-26 Checkpoint](https://drive.google.com/file/d/12I_4qlChWMeigej3KcCtU0JyGI69tOxo/view?usp=sharing).

### ResNet-26 - Results Cifar10C (Level-5 Severity)

|                                                            | data samples used| mean error | gauss_noise | shot_noise | impulse_noise | defocus_blur | glass_blur | motion_blur | zoom_blur | snow | frost |  fog | brightness | contrast | elastic_trans | pixelate | jpeg |
| ---------------------------------------------------------- | ---:|---: | ----------: | ---------: | ------------: | -----------: | ---------: | ----------: | --------: | ---: | ----: | ---: | ---------: | -------: | ------------: | -------: | ---: |
| source        |10000 | 49.2|45.6| 41.8| 50.0| 21.8| 46.1| 23.0| 23.9| 29.9| 30.0| 25.1| 12.2| 23.9| 22.6| 47.2| 27.2|
| norm         |10000 | 33.0 | 44.6| 43.7| 49.1| 29.4| 45.2| 26.2| 26.9| 25.8| 27.9| 23.8| 18.3| 34.3| 29.3| 37.0| 32.5|
| ttt           |10000 | 31.4 | 45.6| 41.8| 50.0| 21.8| 46.1| 23.0| 23.9| 29.9| 30.0| 25.1| 12.2| 23.9| 22.6| 47.2| 27.2|
| dua          |80 |26.8|34.9| 32.6| 42.2| 18.7| 40.2| 24.0| 18.4| 23.9| 24.0| 20.9| 12.3| 27.1| 27.2| 26.2| 28.7|

**Usage**:
```python
python dua.py --model res --dataroot ROOT_PATH_FOR_CIFAR_10C_DATASET --ckpt_path PATH_FOR_DOWNLOADED_CHECKPOINT
```
To cite us: 
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

## Acknowledgments
Part of our code is built upon the public code of [TTT](https://github.com/yueatsprograms/ttt_cifar_release).
