import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from matplotlib import pyplot as plt



class LISC(Dataset):

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
        return f'The LISC Dataset is Selected for {self.for_data}. Number of {self.for_data} Data is: {self.__len__()} '

    def __getitem__(self, idx):
        image_add = os.path.join(self.image_dir, self.file_names[idx]+'.bmp')
        mask_add = os.path.join(self.mask_dir, self.file_names[idx]+'_expert.bmp')
        image = np.asarray(Image.open(image_add).resize(self.target_size))
        mask = np.asarray(Image.open(mask_add))
        if self.num_label==2:
            mask  = np.where((mask!=0),1,0)
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(self.target_size))

        if self.transforms:
            aug_out = self.transforms(image=image, mask=mask)
            image = aug_out['image']
            mask = aug_out['mask']        

        image = image.transpose(2,0,1)
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long)

        data_info = {'image': image,
                      'mask': mask,
                      'name': self.file_names[idx].replace('/', '')
        }
        if self.for_data == 'Test':
            return data_info
        else:
            return image, mask



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
        
        plt.savefig('lisc_showsample.png')
        plt.show()



    def read_files(self):
        file_names = []
        list_dir = os.listdir(self.image_dir)
        for folder_name in list_dir:
            current_add = os.path.join(self.image_dir, folder_name)
            if os.path.isdir(current_add):
                for file_name in os.listdir(current_add):
                    if file_name.endswith('.bmp'):
                        file_add = os.path.join(folder_name, file_name.split('.')[0])
                        file_names.append(file_add)
        return file_names




if __name__ == '__main__':
    my_dataset = LISC(image_dir = '/content/drive/MyDrive/Cell_Segmentation/Data/LISC/Main Dataset',
                      mask_dir = '/content/drive/MyDrive/Cell_Segmentation/Data/LISC/Ground Truth Segmentation',
                      )
    print(my_dataset)
    my_dataset.show_samples(3)
    my_dataloader = DataLoader(my_dataset, shuffle=True)

    # for image, mask in my_dataloader:
    #     print(torch.unique(mask))
    #     print(image.shape)
    #     print(mask.shape)
    #     fig, axs = plt.subplots(1,2)
    #     axs[0].imshow(image.numpy().squeeze().transpose(1,2,0))
    #     axs[1].imshow(mask.numpy().squeeze())
    #     plt.savefig('test.png')

    #     break




