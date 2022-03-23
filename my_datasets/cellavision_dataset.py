import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt



class CELLAVISION(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, file_names=None, target_size=(256,256), for_data='Train', num_label=2):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.target_size = target_size
        self.for_data = for_data
        self.num_label = num_label 
        if file_names is None:
            self.file_names = self.read_files()
        else:
            self.file_names = file_names

    def __len__(self):
        return len(self.file_names)
    
    def __str__(self):
        return f'The CELLAVISION Dataset is Selected for {self.for_data}. Number of {self.for_data} Data is: {self.__len__()} '
        
    def __getitem__(self, idx):
        image_add = os.path.join(self.image_dir, self.file_names[idx]+'.tif')
        mask_add = os.path.join(self.mask_dir, self.file_names[idx]+'.jpg')
        image = np.asarray(Image.open(image_add).resize(self.target_size))
        mask = np.asarray(Image.open(mask_add).convert('L').resize(self.target_size))
        mask = np.where(mask > 180, 1, (np.where(mask < 64, 0, 2)))

        if self.num_label==2:
            mask  = np.where((mask!=0),1,0)

        if self.transforms:
            aug_out = self.transforms(image=image, mask=mask)
            image = aug_out['image']
            mask = aug_out['mask']  


        image = image.transpose(2,0,1)
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long)

        data_info = {'image': image,
                      'mask': mask,
                      'name': self.file_names[idx]
        }
        if self.for_data == 'Test':
            return data_info
        else:
            return image, mask
            

    def read_files(self):
        file_names = []
        images = os.listdir(self.image_dir)
        masks = os.listdir(self.mask_dir)
        for file_name in images:
            if file_name.endswith('.tif'):
                if file_name.split('.')[0] + '.jpg' in masks:
                    file_names.append(file_name.split('.')[0])

        return file_names
  
  
    def show_samples(self, n):
        fig, axs = plt.subplots(nrows=2, ncols=n, figsize= (15,3), squeeze=False)
        for i in range(0,n):
            idx = np.random.randint(low=0, high=self.__len__()-1)
            image, mask = self.__getitem__(idx)
            image = image.numpy().transpose(1,2,0).astype(np.uint8)
            mask = mask.numpy()
            axs[0,i].imshow(image)
            axs[0,i].axis('off')  
            axs[1,i].imshow(mask, cmap='gray')
            axs[1,i].axis('off') 
            if i == n//2:
                axs[0,i].set_title('--------------------------------------------------Images-------------------------------------------------')
                axs[1,i].set_title('--------------------------------------------------Maskes-------------------------------------------------')
        
        plt.savefig('cellavision_showsample.png')
        plt.show()


if __name__ == '__main__':
    ds = CELLAVISION('/content/drive/MyDrive/Cell_Segmentation/Data/CellaVision/IDB1/images', '/content/drive/MyDrive/Cell_Segmentation/Data/CellaVision/IDB1/labels')
    print(ds)
    dl = DataLoader(ds, 1)
    ds.show_samples(5)
