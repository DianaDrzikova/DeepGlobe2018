# DeepGlobe Challenge 2018 - Road Extraction

Training and evaluation of efficientnet-b5 u-net model pre-trained on ImageNet dataset with custom loss function "Center Loss". Center loss combines DiceLoss and MSE in which MSE plays role of soft-thresholder.

Achieved 0.7978 IOU score on last 1000 images in the train split (test split does not provide labels).  

## Set-up:
Install requirements:
```pip install -r requirements.txt```

Download dataset:
```python download_dataset.py```

## Config:
Adjust dataset path in config file based on where you downloaded it.

## Run:
1. Run code in:
-  ```run_clean.ipynb```

2. Run code in terminal
- Training: ```python train.py --config congif.yaml```
- Evaluation: ```python evaluate.py --config congif.yaml```

## Results:
The trained model is in ```checkpoints``` and loss and iou metrics are in ```images```.
