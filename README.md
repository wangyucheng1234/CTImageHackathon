# CTImageHackathon

## Usage

### Generate training slices
1. Download ModMedData CT dataset  https://mosmed.ai/datasets/covid19_1110/
3. Config
```bash
base_path = './'
```
in ```bash preprocess.py```
3. run ```bash python preprocess.py```


### Train model

1.Config ```train.py```
```bash
basepath = './'
train_epoches = 500
val_period = 5
```
2.run ```bach python train.py```

### Test

1.Config ```test.py```
```bash
model_path = 'model/405.model'
test_image_path = 'studies/CT-0/study_0200.nii.gz'
threshold = [0, 3299.032990329903, 47378.47378473785, 86080.86080860808, 214349]
```
2.run ```bach python test.py```
