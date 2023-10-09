from utils import *

class RemoteSensingDataset(Dataset):

    def __init__(self, master_gdf, size=224, labels=True):
        self.master_gdf = master_gdf
        self.size = size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        image = append_index_layers(image)
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
        
