"""Image-only training augmentation. Dense ground truth is inaccessible here."""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import transforms
from research.config import image_path


class ImageLabelDataset(Dataset):
    def __init__(self, data, ids, labels, crop, config=None):
        self.data=Path(data); self.ids=Path(ids).read_text().split(); self.crop=crop
        self.labels=np.load(labels,allow_pickle=True).item()
        self.config=config or {'data': str(self.data)}

    def __len__(self): return len(self.ids)

    def __getitem__(self,i):
        name=self.ids[i]
        image=np.asarray(Image.open(image_path(self.config,name,'train')).convert('RGB'))
        image=transforms.random_scaling(image,scale_range=[.5,2.0])
        if random.random()<.5: image=np.fliplr(image).copy()
        image,box=transforms.random_crop(image,label=None,crop_size=self.crop,mean_rgb=[123.675,116.28,103.53])
        image=transforms.normalize_img(image).transpose(2,0,1).copy()
        # Explicit padding mask, independent of segmentation labels.
        return name,torch.from_numpy(image),torch.tensor(self.labels[name],dtype=torch.float32),torch.tensor(box)


def load_eval_image(data,name,config=None):
    raw=np.asarray(Image.open(image_path(config or {'data': str(data)},name,'val')).convert('RGB'))
    normalized=transforms.normalize_img(raw).transpose(2,0,1).copy()
    return raw,torch.from_numpy(normalized)[None]
