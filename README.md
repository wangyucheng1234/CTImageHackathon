# CTImageHackathon

## Requirement

```bash
torch==1.8.1+cu111
matplotlib==3.4.2
pandas==1.2.4
torchnet==0.0.4
scipy==1.6.3
numpy==1.19.5
scikit_learn==1.0
SimpleITK==2.1.1
tensorboardX==2.4
```

## Usage

### Generate training slices
1. Download ModMedData CT dataset  https://mosmed.ai/datasets/covid19_1110/
2. Config
```bash
base_path = './'
```
in ```preprocess.py```
3. run ```python preprocess.py```


### Train model

1.Config ```train.py```
```bash
basepath = './'
train_epoches = 500
val_period = 5
```
2.run ```python train.py```

You can download our trained weight file here https://drive.google.com/file/d/1GnOtJ4d4CE1vfiG15FYYFgmtr40y-2Dw/view?usp=sharing

### Segment each CT scans and find optimal threshold
1. Config ```Analysis.py```
```bash
basepath = './'
model_path = 'model/405.model'
```
2. run ```python Analysis.py```


### Test

1.Config ```test.py```
```bash
model_path = 'model/405.model'
test_image_path = 'studies/CT-0/study_0200.nii.gz'
threshold = [0, 3299.032990329903, 47378.47378473785, 86080.86080860808, 214349]
```
2.run ```python test.py```
