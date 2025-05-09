import os
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MulanLayerDataset(Dataset):
    def __init__(self, data_folder, resolution=(256,256)):
        self.data_folder = data_folder
        self.resolution = resolution
        self.layer_dirs = [f"layer_{i}" for i in range(6)]
        
        # 1) scan all files and group by base ID
        groups = defaultdict(set)
        for i, d in enumerate(self.layer_dirs):
            folder = os.path.join(data_folder, d)
            for fn in os.listdir(folder):
                if fn.endswith(".png") and "-layer_" in fn:
                    base_id, _ = fn.split("-layer_")
                    groups[base_id].add(i)
        
        # 2) keep only those IDs that have all 6 layers
        self.ids = sorted([bid for bid, layers in groups.items() if layers == set(range(6))])
        
        # 3) prepare transform
        self.transform = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        bid = self.ids[idx]
        layers = []
        for i in range(6):
            fn = f"{bid}-layer_{i}.png"
            path = os.path.join(self.data_folder, f"layer_{i}", fn)
            img = Image.open(path).convert("RGB")
            layers.append(self.transform(img))
        return torch.stack(layers, dim=0)  # shape (6,3,256,256)

# usage example
if __name__=="__main__":
    data_folder = "/work3/s243345/adlcv/project/MULAN_data"
    ds = MulanLayerDataset(data_folder, resolution=(256,256))
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    batch = next(iter(dl))
    print(batch.shape)  # e.g. (8,6,3,256,256)
