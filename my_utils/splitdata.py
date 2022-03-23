import random
import os

def split_data(image_dir, mask_dir, test_p, val_p, dataset_name):
    if dataset_name=='WBC_DATASET1' or dataset_name=='WBC_DATASET2':
        file_names = read_files_wbc(image_dir)
    elif dataset_name=='CELLAVISION_DATASET':
        file_names = read_files_cellavision(image_dir, mask_dir)
    else:
        file_names = read_files_lisc(image_dir)

    num_all_data = len(file_names)

    num_test_data = round(test_p*num_all_data)
    num_val_data = round(val_p*num_all_data)

    test_set = random.sample(file_names, k=num_test_data)
    file_names = Diff(file_names, test_set)

    val_set = random.sample(file_names, k=num_val_data)

    file_names = Diff(file_names, val_set)
    train_set = file_names

    data = {'Train':train_set, 'Validation':val_set, 'Test':test_set}

    return data


    




def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


def read_files_wbc(image_dir):
    file_names = []
    for file_name in os.listdir(image_dir):
        file_names.append(file_name)

    return file_names

def read_files_lisc(image_dir):
    file_names = []
    list_dir = os.listdir(image_dir)
    for folder_name in list_dir:
        current_add = os.path.join(image_dir, folder_name)
        if os.path.isdir(current_add):
            for file_name in os.listdir(current_add):
                if file_name.endswith('.bmp'):
                    file_add = os.path.join(folder_name, file_name.split('.')[0])
                    file_names.append(file_add)
    return file_names


def read_files_cellavision(image_dir, mask_dir):
    file_names = []
    images = os.listdir(image_dir)
    masks = os.listdir(mask_dir)
    for file_name in images:
        if file_name.endswith('.tif'):
            if file_name.split('.')[0] + '.jpg' in masks:
                file_names.append(file_name.split('.')[0])

    return file_names