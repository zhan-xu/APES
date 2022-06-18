This is the code repository implementing the paper "APES: Articulated Part Extraction from Sprite Sheets".
 
## Setup
The project is developed on Ubuntu 20.04 with cuda11 and cudnn8.

```
conda create --name apes python=3.7
conda activate apes
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
pip install scikit-image==0.18.2 tqdm==4.62.3 protobuf==3.20.0 tensorboard==1.15.0 scikit-learn==0.24.2 h5py==3.6.0 opencv-python==4.5.2.54 kornia==0.5.11
pip install --no-index torch-cluster torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install matplotlib imageio plotly
conda install pytorch3d -c pytorch3d
```

## Testing & Evaluation
### Testing Data
Download our testing data from the following links:
1. [OkaySamurai](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/Ef1skomqQAtDh4ABrywdo2oB7fUJSgmBY2dYP-zlxva-0A?e=56V7Yg)
2. [Sprite dataset](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EXFVwyVPLctIq3nPamoxiwAB8e5Mz1eU72zIWAicBWKZBA?e=XJo6O2).
### Pretrained models
Download our pretrained models from [here](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EYv0I4AyNnJEtr9pU_yEv_gB2AbiCYaWMRtCQkKYq9dhRA?e=DzcwN7).
### Testing steps:
1. To test on OkaySamurai data:
```
python -u inference/inference_os.py --init_weight_path="checkpoints/train_cluster/model_best.pth.tar" --test_folder="DATASET_PATH/okaysamurai_sheets/" --output_folder=CUSTOMIZED_OUTPUT_FOLDER
```
2. To qualitatively evaluate on OkaySamurai data, use script `evaluate/evaluate_select.py`. 
Set "data_folder" and "res_folder" to the paths to "okaysamurai_sheets" folder and your output folder.
3. To test on Sprite data:
```
python -u inference/inference_real.py --init_weight_path="checkpoints/train_cluster/model_best.pth.tar" --test_folder="DATASET_PATH/real_sheets/" --output_folder=CUSTOMIZED_OUTPUT_FOLDER
```
## Training
### Training Datasets
Download the training datasets from the following links:
1. [OkaySamurai dataset](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EUk_2ZznigpNjCqr8wx4pBIBDs85cDHPwm4jFgFrc0S3FQ?e=NjbQD0) (1.9G, 34.9G after decompression).

2. (Optional) A [processed subset](https://umass-my.sharepoint.com/:u:/g/personal/zhanxu_umass_edu/EUQNyinpUhxCgp2rRtXTAAkBANRZyxJpKZX2lXTfDU9OmA?e=T5Utnn) of CreativeFlow+ dataset we used to pretrain the correspondence module (992M, 22G after decompression). Original dataset is [here](https://www.cs.toronto.edu/creativeflow/#download).

### Training steps:
1. (Optional) Pretrain the correspondence module on Creative Flow+ dataset.

Download our processed Creative Flow+ dataset from above link, 
then run the following command (change the paths to the place you put the data):
```
python -u training/train_correspondence.py \
--train_folder="creative_flow_folder/train/" \
--val_folder="creative_flow_folder/val/" \
--test_folder="creative_flow_folder/test/" \
--train_batch=8 --test_batch=8 \
--logdir="logs/pretrain_corrnet_cf" \
--checkpoint="checkpoints/pretrain_corrnet_cf" \
--epochs=10 --lr=1e-3 --workers=4 --dataset="creativeflow"
```
2. Pretrain the correspondence module on OkaySamurai dataset.

Download OkaySamurai dataset from the above link. 
Data is organized as a h5 file which has been included in our provided dataset file. It is generated by `data_proc/generate_hdf5.py`. 
Run the following command to train (change the paths to the place you put the data):
```
python -u training/train_correspondence.py \
--train_folder="okay_samurai_folder/train/" \
--val_folder="okay_samurai_folder/val/" \
--test_folder="okay_samurai_folder/test/" \
--train_batch=8 --test_batch=8 \
--logdir="logs/pretrain_corrnet_os" \
--checkpoint="checkpoints/pretrain_corrnet_os" \
--epochs=20 --lr=1e-3 --workers=4 --dataset="okaysamurai" --schedule 5
```
If you've taken step 1, simply add 
```
--init_corrnet_path="checkpoints/pretrain_corrnet_cf/model_best.pth.tar"
```
3. Train the clustering module on OkaySamurai dataset with predicted correspondences. 
You will need to first generate predicted correspondences with `inference\output_pred_corr.py`.
The predicted correspondences will be saved at train/val/test split folders.
Then run the following command to train (change the paths to the place you put the data):
```
python -u training/train_fullnet.py \
--train_folder="okay_samurai_folder/train/" \
--val_folder="okay_samurai_folder/val/" \
--test_folder="okay_samurai_folder/test/" \
--train_batch=8 --test_batch=8 \
--logdir="logs/train_cluster" \
--checkpoint="checkpoints/train_cluster" \
--epochs=10 --lr=1e-4 --workers=4 --schedule 5 \
--init_corrnet_path="checkpoints/pretrain_corrnet_os/model_best.pth.tar" \
--offline_corr
```
4. (Optional) The trained model from above steps ("checkpoints/train_cluster/model_best.pth.tar") can already be used. You can optionally finetune both the correspondence module and clustering module together on OkaySamurai dataset by the following command (this step takes about 1-2 days):
```
python -u training/train_fullnet.py \
--train_folder="/mnt/DATA_LINUX/zhan/okay_samurai/train/" \
--val_folder="/mnt/DATA_LINUX/zhan/okay_samurai/val/" \
--test_folder="/mnt/DATA_LINUX/zhan/okay_samurai/test/" \
--train_batch=6 --test_batch=6 \
--logdir="logs/train_fullnet" \
--checkpoint="checkpoints/train_fullnet" \
--epochs=1 --lr=1e-5 \
--workers=4 \
--init_fullnet_path="checkpoints/train_cluster/model_best.pth.tar"
```