from utils import *

def crop_images(year: int,
                mesh_num: int,
                key_id: int,
                gdf,
                mesh_dir='../data/census/mesh',
                land_dir='../data/Landsat',
                ntl_dir ='../data/VIIRS/GeoTiff',
                all_dir ='../data/all_combined',
                size=224,
                mode=None
               ):
    """
    A function of cropping images with the extent of mesh.

    Args:
    year               : An integer indicating the mesh of census boundary.
    mesh_num           : An integer indicating the number of mesh extent.
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
        key_code = gdf.iloc[key_id]['KEY_CODE']
        geom = [gdf.geometry[key_id]]
    except:
        key_code = gdf['KEY_CODE']
        geom = [gdf.geometry]

    # create 'tmp' directory if it does not exist
    if not os.path.exists(os.path.join(all_dir, 'tmp')):
        os.mkdir(os.path.join(all_dir, 'tmp'))

    # crop Landsat-8 (Daytime image)
    with rasterio.open(os.path.join(land_dir, str(year), str(mesh_num)+'.tif')) as img_lands:
        lands_cropped, lands_transf = mask(img_lands, geom, crop=True, all_touched=True)
        lands_meta = img_lands.meta

    # update the metadata
    lands_meta.update({"driver": "GTiff",
                       "height": lands_cropped.shape[1],
                       "width": lands_cropped.shape[2],
                       "transform": lands_transf})

    # store the cropped Landsat-8/OLI
    with rasterio.open(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'), "w", **lands_meta) as dest:
        dest.write(lands_cropped)

    """
    # crop Suomi NPP/VIIRS-DNS
    with rasterio.open(os.path.join(ntl_dir, 'processed', str(year), str(year)+'.tif')) as img_viirs:
        viirs_cropped, viirs_transf = mask(img_viirs, geom, crop=True, all_touched=True)
        viirs_meta = img_viirs.meta

    # update the metadata
    viirs_meta.update({"driver": "GTiff",
                       "height": viirs_cropped.shape[1],
                       "width": viirs_cropped.shape[2],
                       "transform": viirs_transf})

    # store the cropped Suomi NPP/VIIRS-DNS
    with rasterio.open(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'), "w", **viirs_meta) as dest:
        dest.write(viirs_cropped)
    """

    # open the cropped GeoTIFF files
    xds_lands = rioxarray.open_rasterio(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
    #xds_viirs = rioxarray.open_rasterio(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

    # reproject the Suomi NPP/VIIRS-DNS to Landsat-8/OLI for adjusting resolution
    #xds_viirs = xds_viirs.rio.reproject_match(xds_lands)

    # resize pixels
    #xds_viirs = xds_viirs.rio.reproject('EPSG:4326', shape=(size,size))
    xds_lands = xds_lands.rio.reproject('EPSG:4326', shape=(size,size))

    if mode == 'save':
        # save the reprojected VIIRS into GeoTIFF file
        xds_lands.rio.to_raster(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
        #xds_viirs.rio.to_raster(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

    elif mode == 'return':
        # stack the layers into a single np.ndarray object
        #stacked_layers = np.vstack([xds_lands.values, xds_viirs.values])
        stacked_layers = np.vstack([xds_lands.values.astype(np.float32)])
        
        # delete tmp files
        os.remove(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
        #os.remove(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

        return stacked_layers

class RemoteSensingDataset(Dataset):

    def __init__(self, master_gdf, size=224, labels=True):
        self.master_gdf = master_gdf
        self.size = size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cpu":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
        self.labels = labels

    def __len__(self):
        return len(self.master_gdf)

    def __getitem__(self, idx):

        # obtain the target labeled mesh cell
        gdf = self.master_gdf.iloc[idx]

        # get year and key_id
        year = gdf['year']
        mesh_num = gdf['MESH1_ID']
        key_id = gdf['KEY_CODE']

        # crop the image with the extent of labeled mesh cell and convert them into tensor
        image = crop_images(year=year, mesh_num=mesh_num, key_id=key_id, gdf=gdf, size=self.size, mode='return')
        #image = append_index_layers(image)
        image = torch.tensor(image, requires_grad=True).to(self.device)

        if self.labels:
            # log based on 10 and convert them into tensor
            label = gdf[['0_14', '15_64', 'over_64']]
            label = [math.log10(value +1e-7) for value in label]
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
        self.conv_band2_4 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.conv_band5_7 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn_band2_4 = nn.BatchNorm2d(64)
        self.bn_band5_7 = nn.BatchNorm2d(64)
        self.conv_all = nn.Conv2d(
            in_channels=64 + 64, out_channels=64, kernel_size=1, bias=False
        )
        self.resnet = models.resnet50(pretrained=pretrained).to(self.device)
        self.fc2 = nn.Linear(1000, out_dim)
        self.dropout = nn.Dropout(0.25)

        self.conv_band2_4.load_state_dict(self.resnet.conv1.state_dict())
        self.conv_band5_7.load_state_dict(self.resnet.conv1.state_dict())

    def forward(self, x):
        band2_4 = x[:, 2 - 1 : 4, :, :].flip(dims=[0])
        band5_7 = x[:, 5 - 1 : 7, :, :].flip(dims=[0])

        band2_4 = self.conv_band2_4(band2_4)
        band5_7 = self.conv_band5_7(band5_7)

        band2_4 = F.relu(self.bn_band2_4(band2_4))
        band5_7 = F.relu(self.bn_band5_7(band5_7))

        x = torch.cat((band2_4, band5_7), dim=1)
        x = self.conv_all(x)
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
        x = F.relu(self.fc2(x))

        return x
