# 1470_FinalProject: Fantasy Visualizer

## Minghao Liang, Zhen Ren, Yuanbo Li

Fantasy Visualizer is a Deep Learning models that can help readers quickly summarize novel texts and output as images.

### Environment

Please install the requirements specified in requirements.txt

### Seq2Seq usage

#### 1. Prerequisite

| Pretrained_weights & data | Link |
|--------------------------|-----|
|Seq2Seq pretrained weights| <https://drive.google.com/file/d/1XLy_SL6-rS-2GCyCVgPfSgEnBvksWDzd/view?usp=share_link>|te
| Preprocessed data | <https://drive.google.com/file/d/1JYZfhiKi77iPb9H38bkJYPmqoELr1qgb/view?usp=share_link> |
| Output example | <https://drive.google.com/file/d/1BDTbW2albJfGCVwhBDniFBt4wwAUGpMM/view?usp=share_link>

| Required datasets | Link|
|-------------------| -----|
| Pretrained word vectors | <https://drive.google.com/drive/folders/14YR2knlo7ZB6mYGvdzoIW8zhHFcasvmU?usp=share_link> |
| Dataset (CNN Daily News) | <https://drive.google.com/drive/folders/1tXBI0cBhfA3cTgBFCESwwUMm7KsaXuQU?usp=share_link>|
| Dataset (Smaller) | <https://drive.google.com/drive/folders/1LfaR3fTiZ096Nt_4rCQpp-kRb8sQhHxd?usp=share_link>|

#### 2. Usage

 model = load_model('model_epoch_10.h5',custom_objects={'AttentionLayer': AttentionLayer})

