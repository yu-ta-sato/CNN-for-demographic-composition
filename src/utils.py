# for general purposes
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# for geographic data
import rasterio
import rasterio.plot
from rasterio.mask import mask
import rioxarray

# for processing Landsat
import earthpy.plot as ep


# for torch
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    return device


def append_index_layers(image: np.ndarray):
    """
    A function of appending layers of three indices (NDWI, NDBI, NDVI).

    Args:
        image: A np.ndarray object containing 7 layers of Landsat-8 and 1 layer of Suomi NPP.

    Returns:
        image: A np.ndarray object which additionally contains index layers
    """

    # check whether the image has proper number of layers or not
    assert image.shape[0] == 8

    # store relevant bands
    band_3 = image[2, :, :]
    band_4 = image[3, :, :]
    band_5 = image[4, :, :]
    band_6 = image[5, :, :]

    # calculate indices
    NDBI = (band_6 - band_5) / (band_6 + band_5)
    NDVI = (band_5 - band_4) / (band_5 + band_4)
    NDWI = (band_3 - band_5) / (band_3 + band_5)

    # stack index layers
    index_layers = np.stack([NDBI, NDVI, NDWI], axis=0)

    # append index layers to the original image
    image = np.vstack([image, index_layers])

    return image


def visualise_img(features, labels, figsize=10, fig_num=1):
    """
    A function of vilusalising relevant images as samples.
    Args:
        features
        labels
        figsize
        fig_num

    Returns:
        None
    """

    fig, axes = plt.subplots(fig_num, 5, figsize=(figsize, figsize))
    plt.rcParams.update({"font.size": 9})

    for i, ax in enumerate(axes):
        image = features[i]
        label = labels[i]

        # convert tensor into numpy.ndarray
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()

        if isinstance(label, torch.Tensor):
            label = label.cpu().detach().numpy()

        # plot landsat with band 4, 3, 2 for RGB
        ep.plot_rgb(image, rgb=(3, 2, 1), stretch=True, ax=ax[0])
        ax[0].set_title("RGB: Band 4, 3, 2")

        # plot landsat with band 7, 6, 5 for RGB
        ep.plot_rgb(image, rgb=(6, 5, 4), stretch=True, ax=ax[1])
        ax[1].set_title("RGB: Band 7, 6, 5")

        # plot landsat with NDVI
        ep.plot_rgb(image, rgb=(-3, -2, -1), stretch=True, ax=ax[2])
        ax[2].set_title("RGB: NDBI,NDVI,NDWI")
        ax[2].set_xticks([])
        ax[2].set_yticks([])

        # plot VIIRS
        ax[3].imshow(image[7, :, :], cmap="gray")
        ax[3].set_title("NTL")
        ax[3].set_xticks([])
        ax[3].set_yticks([])

        # plot label
        ax[4].bar(x=["0-14", "15-64", "65-"], height=10**label)
        ax[4].set_title("Population by ages")
        ax[4].set_xticklabels(["Btw. 0-14", "Btw. 15-64", "Over 64"])
        ax[4].set_yscale("log")
        ax[4].set_ylim(10**0, 10**6)
    plt.show()
