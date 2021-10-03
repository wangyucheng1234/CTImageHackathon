import gzip
import os
import glob
import SimpleITK as sitk
import torch
import numpy as np
import random

from sklearn.model_selection import train_test_split
import pandas as pd
import pathlib

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

#Train CSV and Test CSV

#train_pd = pd.DataFrame(columns = ('path', 'label'))
#test_pd = pd.DataFrame(columns = ('path', 'label'))

base_path = './'
path = base_path + 'studies/'
slides_path = base_path + 'slides/'
slices_extension = '.npy'
mask_path = base_path + 'masks/'
img_path = base_path + 'studies/CT-1/'


train_list = []
test_list = []

for i in os.listdir(path):
    if 'CT' in i:
        if 'CT-4' not in i:
            subpath = path+i
            #print(subpath)
            datalist = glob.glob(subpath + '/*.nii.gz', recursive = True)
            num_image = len(datalist)
            #sample 20%
            train_idx, val_idx = train_test_split(datalist, test_size=0.2)
            for idx,s in enumerate(train_idx):
                dict_1 = {'path': s, 'label': i}
                train_list.append(dict_1)
            for idx,s in enumerate(val_idx):
                dict_2 = {'path': s, 'label': i}
                test_list.append(dict_2)

train_pd = pd.DataFrame(train_list)
train_pd.to_csv(path + 'train.csv')
test_pd = pd.DataFrame(test_list)
test_pd.to_csv(path + 'test.csv')



train_list = []
test_list = []
train_mask_list = []
val_mask_list = []


#for i in os.listdir(mask_path):
#if 'CT-1' in i:
#    subpath = mask_path+i
#print(subpath)
datalist = glob.glob(mask_path + '/*.nii.gz', recursive = True)
num_image = len(datalist)
print('num', num_image)
#print(num_image)
#sample 20%
train_idx, val_idx = train_test_split(datalist, test_size=0.2)
for idx,s in enumerate(train_idx):
    #print(s.replace(mask_path, img_path))
    dict_1 = {'mask_path': s, 'image_path': s.replace(mask_path, img_path).replace('_mask.nii.gz','.nii.gz')}
    #print(dict_1)
    train_mask_list.append(dict_1)
for idx,s in enumerate(val_idx):
    #print(s.replace(mask_path, img_path))
    dict_2 = {'mask_path': s, 'image_path': s.replace(mask_path, img_path).replace('_mask.nii.gz','.nii.gz')}
    #print(dict_1)
    val_mask_list.append(dict_2)

train_mask_pd = pd.DataFrame(train_mask_list)
test_mask_pd = pd.DataFrame(val_mask_list)
train_mask_pd.to_csv(path + 'train_mask.csv')
test_mask_pd.to_csv(path + 'val_mask.csv')
#print(train_mask_pd)
#print(test_mask_pd)


#train_mask_csv_path = '/content/gdrive/MyDrive/datahackthon/COVID19_1110/COVID19_1110/train_slices.csv'


total_useful_slides = 0
total_useless_slides = 0
total_ROI_pixel = 0
total_back_pixel = 0

if not os.path.isdir(slides_path):
    os.mkdir(slides_path)
if not os.path.isdir(slides_path + 'train/'):
    os.mkdir(slides_path + 'train/')
if not os.path.isdir(slides_path + 'train/image'):
    os.mkdir(slides_path + 'train/image')
if not os.path.isdir(slides_path + 'train/label'):
    os.mkdir(slides_path + 'train/label')

train_slices_list = []
for i in range(len(train_mask_pd)):
    mask_path = train_mask_pd.loc[i]['mask_path']
    image_path = train_mask_pd.loc[i]['image_path']
    mask = sitk.ReadImage(mask_path)
    image = sitk.ReadImage(image_path)
    #image = sitk.Normalize(image)
    mask_array = sitk.GetArrayFromImage(mask)
    image_array = sitk.GetArrayFromImage(image)
    for i in range(mask_array.shape[0]):
        if mask_array[i].sum() == 0:
            slice_useful = 0
        else:
            slice_useful = 1
        mask_name = pathlib.PurePosixPath(mask_path).name.split('.')[0] + '_' + str(i)
        slices_name = pathlib.PurePosixPath(image_path).name.split('.')[0] + '_' + str(i)
        slices = image_array[i]
        mask_slides = mask_array[i]
        #print(mask_slides)
        #slices_pilimg = Image.fromarray(slices)
        #mask_pilimg = Image.fromarray(mask_slides)
        np.save(slides_path + 'train/image/' + slices_name + slices_extension, slices)
        np.save(slides_path + 'train/label/' + mask_name + slices_extension, mask_slides)
        #slices_pilimg.save(slides_path + 'train/image/' + slices_name + slices_extension)
        #mask_pilimg.save(slides_path + 'train/label/' + mask_name + slices_extension)
        #print(slides_path + 'train/image/' + mask_name + slices_extension)
        #print(slides_path + 'train/label/' + slices_name + slices_extension)
        slices_dict = {'slices_path': slides_path + 'train/image/' + slices_name + slices_extension, 'mask_path':slides_path + 'train/label/' + mask_name + slices_extension, 'useful': slice_useful}
        train_slices_list.append(slices_dict)
        if 1:
            if slice_useful:
                total_useful_slides = total_useful_slides + 1
            else:
                total_useless_slides = total_useless_slides + 1
            if 1:
                total_back_pixel = total_back_pixel + (1-mask_array[i]).sum()
                total_ROI_pixel = total_ROI_pixel + mask_array[i].sum()
print(total_useful_slides)
print(total_useless_slides)
print(total_back_pixel)
print(total_ROI_pixel)
print(total_back_pixel/(total_ROI_pixel + total_back_pixel))
print(total_ROI_pixel/(total_ROI_pixel + total_back_pixel))

total_useful_slides = 0
total_useless_slides = 0
total_ROI_pixel = 0
total_back_pixel = 0

if not os.path.isdir(slides_path + 'val/'):
    os.mkdir(slides_path + 'val/')
if not os.path.isdir(slides_path + 'val/image'):
    os.mkdir(slides_path + 'val/image')
if not os.path.isdir(slides_path + 'val/label'):
    os.mkdir(slides_path + 'val/label')

test_slices_list = []
for i in range(len(test_mask_pd)):
    mask_path = test_mask_pd.loc[i]['mask_path']
    image_path = test_mask_pd.loc[i]['image_path']
    mask = sitk.ReadImage(mask_path)
    image = sitk.ReadImage(image_path)
    #volume = sitk.Normalize(volume)
    mask_array = sitk.GetArrayFromImage(mask)
    image_array = sitk.GetArrayFromImage(image)
    for i in range(mask_array.shape[0]):
        if mask_array[i].sum() == 0:
            slice_useful = 0
        else:
            slice_useful = 1
        mask_name = pathlib.PurePosixPath(mask_path).name.split('.')[0] + '_' + str(i)
        slices_name = pathlib.PurePosixPath(image_path).name.split('.')[0] + '_' + str(i)
        slices = image_array[i]
        mask_slides = mask_array[i]
        np.save(slides_path + 'val/image/' + slices_name + slices_extension, slices)
        np.save(slides_path + 'val/label/' + mask_name + slices_extension, mask_slides)
        #slices_pilimg = Image.fromarray(slices)
        #mask_pilimg = Image.fromarray(mask_slides)
        #slices_pilimg.save(slides_path + 'val/image/' + slices_name + slices_extension)
        #mask_pilimg.save(slides_path + 'val/label/' + mask_name + slices_extension)
        #print(slides_path + 'train/image/' + mask_name + slices_extension)
        #print(slides_path + 'train/label/' + slices_name + slices_extension)
        slices_dict = {'slices_path': slides_path + 'val/image/' + slices_name + slices_extension, 'mask_path':slides_path + 'val/label/' + mask_name + slices_extension, 'useful': slice_useful}
        test_slices_list.append(slices_dict)
        if 1:
            if slice_useful:
                total_useful_slides = total_useful_slides + 1
            else:
                total_useless_slides = total_useless_slides + 1
            if 1:
                total_back_pixel = total_back_pixel + (1-mask_array[i]).sum()
                total_ROI_pixel = total_ROI_pixel + mask_array[i].sum()

        #print(t)
        #print(total_back_pixel + total_ROI_pixel)
        #train_mask_slides_path + ''
        #print(mask_array[i].sum())
        #plt.imshow((mask_array[i])*image_array[i])
        #plt.show()

print(total_useful_slides)
print(total_useless_slides)
print(total_back_pixel)
print(total_ROI_pixel)
print(total_back_pixel/(total_ROI_pixel + total_back_pixel))
print(total_ROI_pixel/(total_ROI_pixel + total_back_pixel))

pd.DataFrame(train_slices_list).to_csv(slides_path + 'train_slices.csv')
pd.DataFrame(test_slices_list).to_csv(slides_path + 'test_slices.csv')