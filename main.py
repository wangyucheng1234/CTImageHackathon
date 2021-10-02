
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
import random

from sklearn.model_selection import train_test_split
import pandas as pd

path = './COVID19_1110/studies/'




#train_pd = pd.DataFrame(train_list)
#train_pd.to_csv(path + 'train.csv')
#test_pd = pd.DataFrame(test_list)
#train_pd.to_csv(path + 'test.csv')

#Mask=>mask_slides
import matplotlib.pyplot as plt
import pathlib
from PIL import Image
import numpy as np










train_set = MosMed(slides_path, split = 'train')
val_set = MosMed(slides_path, split = 'val')

train_loader = torch.utils.data.DataLoader(train_set, 16, shuffle=False, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_set, 16, shuffle=False, drop_last=True)