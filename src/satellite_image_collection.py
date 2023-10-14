from utils import *


def get_ee_mesh_boundary(
    mesh_num: int, mesh_dir="../data/census/mesh"
) -> ee.Geometry.Rectangle:
    """
    A function of getting boundary of mesh data in a format of ee.Geometry.Rectangle

    Args:
      mesh_num       : An integer indicating the mesh of census boundary
      mesh_dir       : The directory of mesh boundary. By default, it is '../data/census/mesh'.

    Returns:
      mesh_boundary  : An object of boundary in ee.Geometry.Rectangle
    """

    # initialise Earth Engine Python API
    try:
        ee.Initialize()

    except:
        ee.Authenticate()
        ee.Initialize()

    # import the vector data of mesh
    mesh_path_shp = os.path.join(
        mesh_dir, str(mesh_num), "MESH0" + str(mesh_num) + ".shp"
    )
    gdf = gpd.read_file(mesh_path_shp)

    # get the coordinates of boundary box
    x_min = gdf.unary_union.bounds[0]
    y_min = gdf.unary_union.bounds[1]
    x_max = gdf.unary_union.bounds[2]
    y_max = gdf.unary_union.bounds[3]

    # generate a mesh boundary
    mesh_boundary = ee.Geometry.Rectangle([[x_min, y_min], [x_max, y_max]])

    return mesh_boundary


def fetch_landsat_img(
    mesh: int,
    start_date: str,
    end_date: str,
    bands: [int],
    landsat_type="LANDSAT/LC08/C02/T1",
    mesh_dir="../data/census/mesh",
):
    """
    A function of fetching images from Landsat-8.

    Args:
      mesh         : An integer indicating the mesh of census boundary.
      start_date   : The starting date of sample in "YYYY-MM-DD" format.
      end_date     : The ending date of sample in "YYYY-MM-DD" format.
      landsat_type : The type of Landsat images. By default, Landsat-8 Collection 2 Tier 1 is selected.
      mesh_dir     : The directory of mesh boundary. By default, it is '../data/census/mesh'.

    Returns:
      task         : An ee.batch.Task object for starting the task
    """

    # initialise Earth Engine Python API
    try:
        ee.Initialize()

    except:
        ee.Authenticate()
        ee.Initialize()

    # get mesh boundary
    mesh_boundary = get_ee_mesh_boundary(mesh, mesh_dir)

    # create a collection of Landsat images
    landsat_collection = ee.ImageCollection(landsat_type).filter(
        ee.Filter.date(start_date, end_date)
    )

    # create a simple composition of images (choosing least noisy pixels)
    landsat_result_img = ee.Algorithms.Landsat.simpleComposite(
        **{"collection": landsat_collection, "asFloat": False}
    )

    # choose the band
    landsat_result_img = landsat_result_img.select(["B" + str(b) for b in bands])

    # create a payload
    payload = {
        "image": landsat_result_img,
        "folder": "{}".format(start_date[:4]),  # year
        "description": "{}".format(mesh),
        "region": mesh_boundary.getInfo()["coordinates"],
        "fileFormat": "GeoTIFF",
        "scale": 30,
    }

    # create a task object
    task = ee.batch.Export.image.toDrive(**payload)

    return task


if __name__ == "__main__":
    mesh_nums = os.listdir("../data/census/mesh")
    mesh_nums = [n for n in mesh_nums if len(n) == 4]

    print("Requesting queries for 2015...")
    for i in mesh_nums:
        task = fetch_landsat_img(
            mesh=i,
            start_date="2015-01-01",
            end_date="2015-12-31",
            bands=[1, 2, 3, 4, 5, 6, 7],
        )
        task.start()

    print("Requesting queries for 2020...")
    for i in mesh_nums:
        task = fetch_landsat_img(
            mesh=i,
            start_date="2020-01-01",
            end_date="2020-12-31",
            bands=[1, 2, 3, 4, 5, 6, 7],
        )
        task.start()

    print("Requesting queries for 2022...")
    for i in mesh_nums:
        task = fetch_landsat_img(
            mesh=i,
            start_date="2022-01-01",
            end_date="2022-12-31",
            bands=[1, 2, 3, 4, 5, 6, 7],
        )
        task.start()
