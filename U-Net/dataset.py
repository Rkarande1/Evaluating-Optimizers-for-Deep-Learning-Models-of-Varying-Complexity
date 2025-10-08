import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class PascalVOCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_ids = []
        with open(split_file, 'r') as f:
            for line in f: # Iterate directly over file object for efficiency
                image_id = line.strip()
                if image_id: 
                    self.image_ids.append(image_id)
        
        # check to ensure image_ids list is not empty after filtering
        if not self.image_ids:
            raise ValueError(f"No valid image IDs found in the split file: {split_file}. "
                             "Please ensure it contains image IDs, one per line, and no empty lines.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_id = self.image_ids[index] # <-- Use img_id from the list
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png") #img_id for mask as well

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long() 

        return image, mask
