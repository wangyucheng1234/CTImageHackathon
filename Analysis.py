import torch
import SimpleITK as sitk
import numpy as np
import os
import glob
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import glob
import os
import pathlib
import matplotlib.pyplot as plt
from dataset import MosMed
from model import UNet
from sklearn.mixture import GaussianMixture


basepath = './'



model_path = 'model/405.model'
model = UNet(1, 2).cuda()
model.load_state_dict(torch.load(model_path))
path = basepath + 'studies/'
predict_mask_path = basepath + 'predict_mask/'
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

#ct0 = []
#ct1 = []
#ct2 = []
#ct3 = []
#ct4 = []

if not os.path.isdir(predict_mask_path):
    os.mkdir(predict_mask_path)

if 1:
    for i in os.listdir(path):
        if 'CT' in i:
            subpath = path+i
            savepath = predict_mask_path + i
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
            #print(savepath)
            for j in glob.glob(subpath + '/*.gz', recursive = True):
                itkimage = sitk.ReadImage(j)
                arrayimage = sitk.GetArrayFromImage(itkimage)
                #img_sum = 0
                mask_class = str(pathlib.PurePosixPath(j).parent).split('/')[-1]
                save_mask_path = predict_mask_path + mask_class + '/' + pathlib.PurePosixPath(j).name.split('.')[0]+'_segmask'+'.nii.gz'
                #print(save_mask_path)
                mask_array = np.zeros(arrayimage.shape)
                #print(arrayimage)
                out = sitk.GetImageFromArray(arrayimage)
                #print(itkimage)
                #print(out)
                if 1:
                    with torch.no_grad():
                        for k in range(arrayimage.shape[0]):
                            #print(i)
                            input_tensor = torch.from_numpy(normalize(arrayimage[k])).unsqueeze(0).unsqueeze(0).cuda()
                            output = model(input_tensor)
                            prob = torch.nn.functional.softmax(output)
                            #[Batch, num_class]
                            predict_cls = torch.argmax(prob, axis = 1)
                            mask_array[k] = predict_cls.data.cpu().numpy()[0]
                            #img_sum = img_sum + predict_cls.data.cpu().numpy()[0].sum()
                            #plt.imshow(predict_cls.data.cpu().numpy()[0])
                            #plt.show()
                    out = sitk.GetImageFromArray(mask_array.astype(np.uint8))
                    out.SetDirection(itkimage.GetDirection())
                    out.SetSpacing(itkimage.GetSpacing())
                    out.SetOrigin(itkimage.GetOrigin())
                    sitk.WriteImage(sitk.Cast(out, sitk.sitkUInt8), save_mask_path)
                    #print('sum', img_sum)
                    #print(j)
#load state dict


path = predict_mask_path
volume_list = []

for i in os.listdir(path):
    for j in glob.glob(path + i + '/*.gz', recursive = True):
        #volume_list
        image_name = pathlib.PurePosixPath(j).name
        #print(image_name)
        img = sitk.ReadImage(j)
        img_array = sitk.GetArrayFromImage(img)
        volume_dict = {'name': image_name,'class': i, 'volume':np.sum(img_array)}
        volume_list.append(volume_dict)

volume_pd = pd.DataFrame(volume_list)
volume_pd.to_csv(path + 'volume.csv')


#GaussianMixture(n_component = 4)

CT0 = volume_pd.loc[lambda df: df['class'] == 'CT-0']['volume'].to_numpy()
CT1 = volume_pd.loc[lambda df: df['class'] == 'CT-1']['volume'].to_numpy()
CT2 = volume_pd.loc[lambda df: df['class'] == 'CT-2']['volume'].to_numpy()
CT3 = volume_pd.loc[lambda df: df['class'] == 'CT-3']['volume'].to_numpy()
CT4 = volume_pd.loc[lambda df: df['class'] == 'CT-4']['volume'].to_numpy()

weight0 = len(CT0)/len(volume_pd)
weight1 = len(CT1)/len(volume_pd)
weight2 = len(CT2)/len(volume_pd)
weight3 = len(CT3)/len(volume_pd)
weight4 = len(CT4)/len(volume_pd)

gauss0 = GaussianMixture(n_components=1).fit(CT0[:,np.newaxis])
gauss1 = GaussianMixture(n_components=1).fit(CT1[:,np.newaxis])
gauss2 = GaussianMixture(n_components=1).fit(CT2[:,np.newaxis])
gauss3 = GaussianMixture(n_components=1).fit(CT3[:,np.newaxis])
gauss4 = GaussianMixture(n_components=1).fit(CT4[:,np.newaxis])

import scipy.stats as stats
mu0 = gauss0.means_[0]
sigma0 = np.sqrt(gauss0.covariances_)[0,0]
mu1 = gauss1.means_[0]
sigma1 = np.sqrt(gauss1.covariances_)[0,0]
mu2 = gauss2.means_[0]
sigma2 = np.sqrt(gauss2.covariances_)[0,0]
mu3 = gauss3.means_[0]
sigma3 = np.sqrt(gauss3.covariances_)[0,0]
mu4 = gauss4.means_[0]
sigma4 = np.sqrt(gauss4.covariances_)[0,0]

x = np.linspace(0, 100000, 100000)

plt.plot(x, weight0*stats.norm.pdf(x, mu0, sigma0), label = 'CT-0')
plt.plot(x, weight1*stats.norm.pdf(x, mu1, sigma1), label = 'CT-1')
plt.plot(x, weight2*stats.norm.pdf(x, mu2, sigma2), label = 'CT-2')
plt.plot(x, weight3*stats.norm.pdf(x, mu3, sigma3), label = 'CT-3')
plt.xlabel("Number of pixels")
plt.ylabel("Unnormalized probability of each class")
plt.legend()


#plt.plot(x, stats.norm.pdf(x, mu4, sigma4), label = '4')

plt.show()

print('CT-0',mu0, sigma0)
print('CT-1',mu1, sigma1)
print('CT-2',mu2, sigma2)
print('CT-3',mu3, sigma3)
#gauss0.covariances_

x = np.linspace(0, 100000, 100000)
threshold = []
threshold.append(0)
threshold.append(x[weight0*stats.norm.pdf(x, mu0, sigma0) <weight1*stats.norm.pdf(x, mu1, sigma1)][0])
threshold.append(x[weight1*stats.norm.pdf(x, mu1, sigma1) <weight2*stats.norm.pdf(x, mu2, sigma2)][0])
threshold.append(x[weight2*stats.norm.pdf(x, mu2, sigma2) <weight3*stats.norm.pdf(x, mu3, sigma3)][0])
threshold.append(max(volume_pd['volume']))

print('threshold',threshold)
correct = sum(CT0<threshold[1]) + sum((CT1<threshold[2]) & (CT1>threshold[1])) + sum((CT2<threshold[3]) & (CT2>threshold[2])) + sum((CT3<threshold[4]) & (CT3>threshold[3]))
print('accuracy',correct/(1100 - len(CT4)))