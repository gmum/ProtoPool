# ProtoPool: Interpretable Image  Classification with Differentiable Prototypes Assignment

### Code to a paper published at ECCV 2022 by D. Rymarczyk, Ł. Struski, M. Górszczak, K. Lewandowska, J. Tabor, B. Zieliński 

The code is based on the other repositories: https://github.com/cfchen-duke/ProtoPNet, https://github.com/M-Nauta/ProtoTree and https://github.com/gmum/ProtoPShare


To reproduce results of the paper you need to prepere an environment that meets the requirements from environment.yaml, than prepare data for training, download the pretrained on iNaturalist ResNet50 and run the model. 

Use Python 3.9 with packages in environment.yaml. You can use a following command to install it: conda env create -f environment.yml

## Data preparation:
1. Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. Unpack.
3. Crop the images using information from bounding_boxes.txt (included in the dataset)
4. Split the cropped images into training and test sets, using train_test_split.txt (included in the dataset)
5. Put the cropped training images in the directory "./datasets/birds/train_cropped/"
6. Put the cropped test images in the directory "./datasets/birds/test_cropped/"
7. Augment the training set using img_aug.py (from https://github.com/cfchen-duke/ProtoPNet)
   -- this will create an augmented training set in the following directory:
      "./datasets/birds/train_cropped_augmented/"

## Download pretrained on iNaturalist ResNet50:
1. https://drive.google.com/drive/folders/1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU (Filename on Google Drive: BBN.iNaturalist2017.res50.180epoch.best_model.pth) the model is from (https://github.com/M-Nauta/ProtoTree)
2. Save it in your home directory as 'resnet50_iNaturalist.pth'

## Train a model:
1. for CUB-200-2011: python /app/main.py --data_type birds --num_classes 200 --batch_size 80 --lr "$lr" --epochs "$epochs" --num_descriptive 10 --num_prototypes 202 --results "$save_results" --earlyStopping "$earlyStopping" --use_scheduler --arch resnet50 --pretrained --proto_depth 256 --warmup_time 10 --warmup  --prototype_activation_function log --top_n_weight 0 --last_layer --use_thresh --mixup_data --pp_ortho --pp_gumbel --gumbel_time 30 --inat --data_train [train_dir] --data_push [push_dir] --data_test [test_dir]
2. for Stanford Cars:  python /app/main.py --data_type cars --num_classes 196 --batch_size 80 --lr "$lr" --epochs "$epochs" --num_descriptive 10 --num_prototypes 195 --results "$save_results" --earlyStopping "$earlyStopping" --use_scheduler --arch resnet50 --pretrained --proto_depth 128 --warmup_time 10 --warmup  --prototype_activation_function log --top_n_weight 0 --last_layer --use_thresh --mixup_data --pp_ortho --pp_gumbel --gumbel_time 30 --data_train [train_dir] --data_push [push_dir] --data_test [test_dir]

## For other experiments:
1. To reproduce the influence of prototypes number change the values of --num_prototypes option
2. To reproduce without Gumbel-Softmax run remove --pp_gumbel --gumbel_time 30 options
3. To reproduce results without orthogonalization loss run without -pp_ortho option

The prototype projection process is in the main.py process. So to reproduce those results, you need to check the logs. 

## Citation
```
@inproceedings{rymarczyk2022interpretable,
  title={Interpretable image classification with differentiable prototypes assignment},
  author={Rymarczyk, Dawid and Struski, {\L}ukasz and G{\'o}rszczak, Micha{\l} and Lewandowska, Koryna and Tabor, Jacek and Zieli{\'n}ski, Bartosz},
  booktitle={European Conference on Computer Vision},
  pages={351--368},
  year={2022},
  organization={Springer}
}
```
