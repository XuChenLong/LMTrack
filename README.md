# LMTrack
The official implementation for the **AAAI 2025** paper [_Less Is More: Token Context-Aware Learning for Object Tracking_].

## Install the environment
```
conda create -n lmtrack python=3.8
conda activate lmtrack
bash install.sh
```


## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py \
--script lmtrack --config baseline \
--save_dir ./output \
--mode multiple --nproc_per_node 4 \
--use_wandb 1
```

Replace `--config` with the desired model config under `experiments/lmtrack`.

We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Test and Evaluation

- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py lmtrack baseline --dataset lasot --runid 300 --threads 8 --num_gpus 2
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py lmtrack baseline_384 --dataset got10k_test  --runid 100 --threads 8 --num_gpus 2
python lib/test/utils/transform_got10k.py --tracker_name lmtrack --cfg_name baseline_384
```
- TrackingNet
```
python tracking/test.py lmtrack baseline_fullx4_384 --dataset trackingnet  --runid 300 --threads 8 --num_gpus 2
python lib/test/utils/transform_trackingnet.py --tracker_name lmtrack --cfg_name baseline_fullx4_384
```
