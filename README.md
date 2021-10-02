# CTImageHackathon

## Usage

### Generate training slices

1. Download ModMedData CT dataset  https://mosmed.ai/datasets/covid19_1110/
2. Change
```bash
path = './COVID19_1110/studies/'
base_path = './COVID19_1110/'
slides_path = './COVID19_1110/slides/'
slices_extension = '.npy'
mask_path = './COVID19_1110/masks/'
img_path = './COVID19_1110/studies/CT-1/'
```
in ```bash preprocess.py```
3. run ```bash python preprocess.py```


### Train model

1.Config ```bach train.py```
```bash
train_epoches = 500
val_period = 5
SummaryWriterPath = 'datahackthon/log/segmentation'
SaveModelPath = 'datahackthon/model'
slides_path = './COVID19_1110/slides/'
```
2.run ```bach python train.py```

### Test

1.Config ```bach test.py```
```bash
model_path = './datahackthon/model/405.model'
test_image_path = ''
```
2.run ```bach python test.py```
