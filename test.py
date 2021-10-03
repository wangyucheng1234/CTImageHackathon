import glob
import os
import SimpleITK as sitk
import pathlib
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dataset import MosMed
from model import UNet
import numpy as np
from torchnet import meter
from tensorboardX import SummaryWriter
import random


model_path = 'model/405.model'
test_image_path = 'studies/CT-0/study_0200.nii.gz'
threshold = [0, 3299.032990329903, 47378.47378473785, 86080.86080860808, 214349]
model = UNet(1, 2).cuda()
model.load_state_dict(torch.load(model_path))
#path = './COVID19_1110/studies/'
#predict_mask_path = './COVID19_1110/predict_mask/'
import matplotlib.pyplot as plt
model.eval()

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


#if not os.path.isdir(predict_mask_path):
#    os.mkdir(predict_mask_path)

#if 1:
#    for i in os.listdir(path):
#        if 'CT' in i:
#            subpath = path+i
#            savepath = predict_mask_path + i
#            if not os.path.isdir(savepath):
#                os.mkdir(savepath)
            #print(savepath)
#            for j in glob.glob(subpath + '/*.gz', recursive = True):
itkimage = sitk.ReadImage(test_image_path)
arrayimage = sitk.GetArrayFromImage(itkimage)
#img_sum = 0
#mask_class = str(pathlib.PurePosixPath(j).parent).split('/')[-1]
#save_mask_path = predict_mask_path + mask_class + '/' + pathlib.PurePosixPath(j).name.split('.')[0]+'_segmask'+'.nii.gz'
#print(save_mask_path)
mask_array = np.zeros(arrayimage.shape)
#print(arrayimage)
out = sitk.GetImageFromArray(arrayimage)
#print(itkimage)
#print(out)


sum_pixels = 0

if 1:
    with torch.no_grad():
        for k in range(arrayimage.shape[0]):
            #print(i)
            input_tensor = torch.from_numpy(normalize(arrayimage[k])).unsqueeze(0).unsqueeze(0).cuda()
            output = model(input_tensor)
            prob = torch.nn.functional.softmax(output)
            predict_cls = torch.argmax(prob, axis = 1)
            mask_array[k] = predict_cls.data.cpu().numpy()[0]

    if (np.sum(mask_array)) < threshold[1]:
        print('CT-0')
    elif (np.sum(mask_array))< threshold[2]:
        print('CT-1')
    elif (np.sum(mask_array))< threshold[3]:
        print('CT-2')
    else:
        print('CT-3')

#out = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    #out.SetDirection(itkimage.GetDirection())
    #out.SetSpacing(itkimage.GetSpacing())
    #out.SetOrigin(itkimage.GetOrigin())
    #sitk.WriteImage(sitk.Cast(out, sitk.sitkUInt8), save_mask_path)

