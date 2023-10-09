from utils import *

def preprocess_viirs(master_gdf,
                     year: int,
                     mesh_dir='../data/census/mesh',
                     ntl_dir ='../data/VIIRS/GeoTiff'):
    """
    A function of pre-processing VIIRS images.

    Args:
        master_gdf :
        year       : An integer indicating the mesh of census boundary.
        mesh_dir   : The directory of mesh boundary. By default, it is '../data/census/mesh'.
        ntl_dir    : The directory of VIIRS images. By default, it is ='../data/VIIRS/GeoTiff'.

    Returns:
        None
    """

    # import VIIRS
    img = rasterio.open(os.path.join(ntl_dir, 'original', str(year), str(year)+'.tif'))

    # mask VIIRS with master_gdf
    output, transform = mask(img, master_gdf.geometry, crop=True, all_touched=True)

    # set negative value as zero
    output = np.where(output < 0, 0, output)

    # take log
    output = np.log(output + 1e-7)

    # get the metadata
    meta = img.meta

    # update the metadata
    meta.update({"driver": "GTiff",
                 "height": output.shape[1],
                 "width": output.shape[2],
                 "transform": transform})

    # save the cropped and normalized image
    with rasterio.open(os.path.join(ntl_dir, 'processed', str(year), str(year)+'.tif'), "w", **meta) as dest:
        dest.write(output)

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

    # open the cropped GeoTIFF files
    xds_lands = rioxarray.open_rasterio(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
    xds_viirs = rioxarray.open_rasterio(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

    # reproject the Suomi NPP/VIIRS-DNS to Landsat-8/OLI for adjusting resolution
    xds_viirs = xds_viirs.rio.reproject_match(xds_lands)

    # resize pixels
    xds_viirs = xds_viirs.rio.reproject('EPSG:4326', shape=(size,size))
    xds_lands = xds_lands.rio.reproject('EPSG:4326', shape=(size,size))

    if mode == 'save':
        # save the reprojected VIIRS into GeoTIFF file
        xds_lands.rio.to_raster(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
        xds_viirs.rio.to_raster(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

    elif mode == 'return':
        # stack the layers into a single np.ndarray object
        stacked_layers = np.vstack([xds_lands.values, xds_viirs.values])

        # delete tmp files
        os.remove(os.path.join(all_dir, 'tmp', str(key_code) + '_landsat.tif'))
        os.remove(os.path.join(all_dir, 'tmp', str(key_code) + '_viirs.tif'))

        return stacked_layers

if __name__ == "__main__":
    master_gdf = gpd.read_file('../data/census/master_gdf.gpkg')

    for year in [2015,2020,2022]:
        preprocess_viirs(master_gdf, year)

