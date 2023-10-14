from utils import *


def get_labeled_mesh(
    year: int, mesh_num: int, mesh_dir: str
) -> gpd.geodataframe.GeoDataFrame:
    """
    A function of fetching aggregated mesh with label of demographics.

    Args:
      year          : An integer indicating the mesh of census boundary.
      mesh_num      : An integer indicating the number of mesh extent.
      mesh_dir      : A string indicating the directory of mesh boundary.

    Returns:
      gdf          : A GeoDataDrame object which contains vector data of aggregated mesh and demographics.
    """

    # get the path of mesh data
    mesh_path_shp = os.path.join(
        mesh_dir, str(mesh_num), "MESH0" + str(mesh_num) + ".shp"
    )

    # import and clean up the mesh data
    gdf = gpd.read_file(mesh_path_shp)
    gdf["KEY_CODE"] = gdf["KEY_CODE"].astype(int)

    # create a dictionary for storing the prefix of file name by year
    prefix_dict = {2015: "tblT000846S", 2020: "tblT001100S"}

    # get the path of census table
    table_path = os.path.join(
        "../data/census/population",
        str(year),
        prefix_dict[year] + str(mesh_num) + ".txt",
    )

    # return None if there is no data available for the extent
    if not os.path.exists(table_path):
        return None

    else:
        # import and clean up the census table
        pop_df = pd.read_csv(table_path, encoding="unicode_escape")
        pop_df = pop_df.iloc[1:,]
        pop_df = pop_df.fillna(0).replace("*", 0)
        pop_cols = [c for c in pop_df.columns if c.startswith("T00")]
        pop_df[pop_cols] = pop_df[pop_cols].astype(float)
        pop_df["KEY_CODE"] = pop_df["KEY_CODE"].astype(int)

        # merge the mesh data with the census table
        gdf = gdf.merge(pop_df, how="left", on="KEY_CODE")

        # fulfill NA with zero
        gdf = gdf.fillna(0)

        # identify the columns to be used
        if year == 2015:
            col_list = ["T000846004", "T000846010", "T000846016"]

        elif year == 2020:
            col_list = ["T001100004", "T001100010", "T001100019"]

        # change name of columns
        gdf[["0_14", "15_64", "over_64"]] = gdf[col_list]

        # select only relevant columns
        gdf = gdf[
            ["MESH1_ID", "MESH2_ID", "KEY_CODE", "0_14", "15_64", "over_64", "geometry"]
        ]

        # aggregate by the id of broader extent
        gdf = gdf.dissolve(
            by=["MESH1_ID", "MESH2_ID"], aggfunc="sum", numeric_only=False
        )

        # add a column of year
        gdf["year"] = year

        # remove tile with zero population
        gdf = gdf[np.sum(gdf[["0_14", "15_64", "over_64"]], axis=1) != 0]

        return gdf


def main():
    mesh_nums = os.listdir("../data/census/mesh")
    mesh_nums = [n for n in mesh_nums if len(n) == 4]
    mesh_dir = "../data/census/mesh"

    master_gdf = None
    for year in [2015, 2020]:
        print("Meshes being aggregated for {}...".format(year))

        for mesh_num in mesh_nums:
            gdf = get_labeled_mesh(year, mesh_num, mesh_dir)
            master_gdf = pd.concat([master_gdf, gdf])

    master_gdf.to_file("../data/census/master_gdf.gpkg", driver="GPKG", layer="name")
    print("Mesh aggregation completed.")

    return master_gdf


if __name__ == "__main__":
    main()
