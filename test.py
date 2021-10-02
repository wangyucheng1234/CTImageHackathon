import glob
import os
import SimpleITK as sitk
import pathlib
import torch
import numpy as np


model_path = './datahackthon/model/405.model'
model = UNet(1, 2).cuda()
model.load_state_dict(torch.load(model_path))
path = './COVID19_1110/studies/'
predict_mask_path = './COVID19_1110/predict_mask/'
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
                #out = sitk.GetImageFromArray(arrayimage)
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
                    if ((np.sum(mask_array)) <100) and ('CT-0' not in i):
                        print(j)
                    elif (np.sum(mask_array) > 500) and ('CT-0' in i):
                        print(j)
                    #out = sitk.GetImageFromArray(mask_array.astype(np.uint8))
                    #out.SetDirection(itkimage.GetDirection())
                    #out.SetSpacing(itkimage.GetSpacing())
                    #out.SetOrigin(itkimage.GetOrigin())
                    #sitk.WriteImage(sitk.Cast(out, sitk.sitkUInt8), save_mask_path)
                    #print('sum', img_sum)
                    #print(j)
