# Contrastive Learning for Object Detection

This repository builds upon [detectron2](https://github.com/facebookresearch/detectron2) and [RINCE](https://github.com/boschresearch/rince). Project for AI 535 and AI 537 at Oregon State University

## Setting up the repository
Clone the repo

```
git clone https://github.com/rishabbala/Contrastive_Learning_For_Object_Detection
```

Make sure to download the CIFAR10 and VOC datasets into the datasets folder
Or alternatively set
```
...download=True
```
in the dataloader

## Training on CIFAR10
```
python main_supcon.py --batch_size 512 --num_workers 16 --print_freq 20 --data_folder ./datasets --dataset cifar10 cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.6 --similarity_threshold 0.5 --n_sim_classes 1 --use_supercategories True --use_same_and_similar_class True --mixed_out_in_log True --exp_name CIFAR10_baseline --test False --map False --tsne False --save_fig False --config-file ./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl
```

## Training on VOC
```
python main_supcon.py --batch_size 32 --num_workers 16 --print_freq 20 --data_folder ./datasets --dataset voc cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.6 --similarity_threshold 0.5 --n_sim_classes 1 --use_supercategories True --use_same_and_similar_class True --mixed_out_in_log True --exp_name CIFAR10_baseline --test True --map False --tsne False --save_fig False --config-file ./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl
```

## Testing on CIFAR10
```
python main_supcon.py --batch_size 512 --num_workers 16 --print_freq 20 --data_folder ./datasets --dataset cifar10 --cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.6 --similarity_threshold 0.5 --n_sim_classes 1 --use_supercategories True --use_same_and_similar_class True --mixed_out_in_log True --exp_name CIFAR10_baseline --test True --map False --tsne False --save_fig False --checkpoint $(your_checkpoint_location) --config-file ./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl
```

## Testing on VOC
```
python main_supcon.py --batch_size 32 --num_workers 16 --print_freq 20 --data_folder ./datasets --dataset voc --cosine --learning_rate 0.5 --min_tau 0.1 --max_tau 0.6 --similarity_threshold 0.5 --n_sim_classes 1 --use_supercategories True --use_same_and_similar_class True --mixed_out_in_log True --exp_name CIFAR10_baseline --test True --map False --tsne False --save_fig False --checkpoint $(your_checkpoint_location) --config-file ./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl
```

Similarly, run out_of_dist_detection.py for OOD Detection. baseline.py and baseline_ood.py use a softmax classification
