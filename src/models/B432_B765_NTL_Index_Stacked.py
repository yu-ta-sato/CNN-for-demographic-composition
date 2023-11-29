import os
import math
import numpy as np
import rasterio
from rasterio.mask import mask
import rioxarray

# for torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset

import warnings


def crop_images(
    year: int,
    mesh_num: int,
    key_id: int,
    gdf,
    land_dir="../data/Landsat",
    ntl_dir="../data/VIIRS/GeoTiff",
    all_dir="../data/all_combined",
    size=224,
    mode=None,
):
    """
    A function of cropping images with the extent of mesh.

    Args:
    year               : An integer indicating the mesh of census boundary.
    key_id             : An integer indicating the id of mesh cell.
    gdf                : An GeoDataDrame object which contains vector data of aggregated mesh and demographics.
    mesh_dir           : The directory of mesh boundary. By default, it is '../data/census/mesh'.
    land_dir           : The directory of landsat images. By default, it is '../data/Landsat'.
    ntl_dir            : The directory of VIIRS images. By default, it is '../data/VIIRS/GeoTiff'.
    all_dir            : The directory of final output. By default, it is '../data/all_combined'.
    size               : An integer of final output pixel size. By default, it is 224.
    mode               : An string indicating either 'save' or 'return' the output.

    Returns:
        stacked_layers : A np.ndarray object of stacked layers from Landsat-8 and VIIRS.
    """

    # obtain the geometry of mesh cell
    try:
        key_code = gdf.iloc[key_id]["KEY_CODE"]
        geom = [gdf.geometry[key_id]]
    except:
        key_code = gdf["KEY_CODE"]
        geom = [gdf.geometry]

    # create 'tmp' directory if it does not exist
    if not os.path.exists(os.path.join(all_dir, "tmp")):
        os.mkdir(os.path.join(all_dir, "tmp"))

    # crop Landsat-8 (Daytime image)
    with rasterio.open(
        os.path.join(land_dir, str(year), str(mesh_num) + ".tif")
    ) as img_lands:
        lands_cropped, lands_transf = mask(img_lands, geom, crop=True, all_touched=True)
        lands_meta = img_lands.meta

    # update the metadata
    lands_meta.update(
        {
            "driver": "GTiff",
            "height": lands_cropped.shape[1],
            "width": lands_cropped.shape[2],
            "transform": lands_transf,
        }
    )

    # store the cropped Landsat-8/OLI
    with rasterio.open(
        os.path.join(all_dir, "tmp", str(key_code) + "_landsat.tif"), "w", **lands_meta
    ) as dest:
        dest.write(lands_cropped)

    # crop Suomi NPP/VIIRS-DNS
    with rasterio.open(
        os.path.join(ntl_dir, "processed", str(year), str(year) + ".tif")
    ) as img_viirs:
        viirs_cropped, viirs_transf = mask(img_viirs, geom, crop=True, all_touched=True)
        viirs_meta = img_viirs.meta

    # update the metadata
    viirs_meta.update(
        {
            "driver": "GTiff",
            "height": viirs_cropped.shape[1],
            "width": viirs_cropped.shape[2],
            "transform": viirs_transf,
        }
    )

    # store the cropped Suomi NPP/VIIRS-DNS
    with rasterio.open(
        os.path.join(all_dir, "tmp", str(key_code) + "_viirs.tif"), "w", **viirs_meta
    ) as dest:
        dest.write(viirs_cropped)

    # open the cropped GeoTIFF files
    xds_lands = rioxarray.open_rasterio(
        os.path.join(all_dir, "tmp", str(key_code) + "_landsat.tif")
    )
    xds_viirs = rioxarray.open_rasterio(
        os.path.join(all_dir, "tmp", str(key_code) + "_viirs.tif")
    )

    # reproject the Suomi NPP/VIIRS-DNS to Landsat-8/OLI for adjusting resolution
    xds_viirs = xds_viirs.rio.reproject_match(xds_lands)

    # resize pixels
    xds_viirs = xds_viirs.rio.reproject("EPSG:4326", shape=(size, size))
    xds_lands = xds_lands.rio.reproject("EPSG:4326", shape=(size, size))

    if mode == "save":
        # save the reprojected VIIRS into GeoTIFF file
        xds_lands.rio.to_raster(
            os.path.join(all_dir, "tmp", str(key_code) + "_landsat.tif")
        )
        xds_viirs.rio.to_raster(
            os.path.join(all_dir, "tmp", str(key_code) + "_viirs.tif")
        )

    elif mode == "return":
        # stack the layers into a single np.ndarray object
        stacked_layers = np.vstack([xds_lands.values, xds_viirs.values])
        # stacked_layers = np.vstack([xds_lands.values.astype(np.float32)])

        # delete tmp files
        os.remove(os.path.join(all_dir, "tmp", str(key_code) + "_landsat.tif"))
        os.remove(os.path.join(all_dir, "tmp", str(key_code) + "_viirs.tif"))

        return stacked_layers


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


class RemoteSensingDataset(Dataset):
    def __init__(self, master_gdf, size=224, labels=True):
        self.master_gdf = master_gdf
        self.size = size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cpu":
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )

        self.labels = labels

    def __len__(self):
        return len(self.master_gdf)

    def __getitem__(self, idx):
        # obtain the target labeled mesh cell
        gdf = self.master_gdf.iloc[idx]

        # get year and key_id
        year = gdf["year"]
        mesh_num = gdf["MESH1_ID"]
        key_id = gdf["KEY_CODE"]

        # crop the image with the extent of labeled mesh cell and convert them into tensor
        image = crop_images(
            year=year,
            mesh_num=mesh_num,
            key_id=key_id,
            gdf=gdf,
            size=self.size,
            mode="return",
        )
        image = append_index_layers(image)
        image = torch.tensor(image, requires_grad=True).to(self.device)

        if self.labels:
            # log based on 10 and convert them into tensor
            label = gdf[["0_14", "15_64", "over_64"]]
            label = [np.log(value) for value in label]
            label = np.maximum(label, 0)

            label = torch.tensor(label, requires_grad=True).float().to(self.device)

            return image, label

        else:
            return image


class RemoteSensingCNN(nn.Module):
    """
    A class of CNN model with transfer learning from ResNet50.
    """

    def __init__(self, out_dim, pretrained):
        super(RemoteSensingCNN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv_input = nn.Conv2d(
            in_channels=12,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn_input = nn.BatchNorm2d(64)
        self.resnet = models.resnet50(pretrained=pretrained).to(self.device)
        self.fc2 = nn.Linear(1000, out_dim)
        self.dropout = nn.Dropout(0.25)

        new_conv1_weights = torch.cat([self.resnet.conv1.weight.data] * 4, dim=1)
        self.conv_input.weight = nn.Parameter(new_conv1_weights)

    def forward(self, x):
        band2_4 = x[:, 2 - 1 : 4, :, :].flip(dims=[0])
        band5_7 = x[:, 5 - 1 : 7, :, :].flip(dims=[0])
        viirs = x[:, 8 - 1, :, :].unsqueeze(dim=1).repeat(1, 3, 1, 1)
        indices = x[:, 9 - 1 : 11, :, :]
        x = torch.cat([band2_4, band5_7, viirs, indices], dim=1)

        x = self.conv_input(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.resnet.fc(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
