# PyTorch implementation of GeoPointGAN

![GeoPointGAN generated and privatized data: 311 caller locations in New York.](https://raw.githubusercontent.com/konstantinklemmer/geopointgan/main/images/gen_new_york.gif)

*(GeoPointGAN generated and privatized data: 311 caller locations in New York.)*

This is the official repository for the paper [GeoPointGAN: Synthetic Spatial Data with Local Label Differential Privacy](https://arxiv.org/abs/2205.08886). *GeoPointGAN* is a generative model for geographic point coordinates and also includes a privacy mechanism.

![GeoPointGAN pipline, including privacy mechanism.](https://raw.githubusercontent.com/konstantinklemmer/geopointgan/main/images/pipeline.png)

*(GeoPointGAN pipline, including privacy mechanism.)*

## Structure

The source code for *GeoPointGAN* can be accessed in the `src` folder, our training datasets (those that are not hosted externally) can be found in the `data` folder and the `notebooks` folder contains an example notebook.

You can open the interactive example notebook straight away via Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/konstantinklemmer/geopointgan/blob/master/notebooks/example.ipynb)

## Citation

If you want to cite our work, you can use the following reference:

``
@misc{cunningham2022geopointgan,
    title={GeoPointGAN: Synthetic Spatial Data with Local Label Differential Privacy},
    author={Teddy Cunningham and Konstantin Klemmer and Hongkai Wen and Hakan Ferhatosmanoglu},
    year={2022},
    eprint={2205.08886},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
``