# MDProp training for CUB200 data with Multisimilarity loss
python main.py --loss multisimilarity --dataset cub200 --source_path /path/to/data --n_epochs 125 --batch_mining distance  --attacker1_targets 1 --attacker2_targets 5 --arch resnet50_normalize --bs 64 --gpu 1,2,3 --seed 3

# MDProp training for CUB200 data with S2SD
python main.py --loss s2sd --dataset cub200 --source_path /path/to/data --n_epochs 125 --bs 64  --attacker1_targets 1 --attacker2_targets 5 --seed 1 --gpu 1,2,3 --arch resnet50_normalize --embed_dim 128 --loss_s2sd_source multisimilarity --loss_s2sd_target multisimilarity --loss_s2sd_T 1 --loss_s2sd_w 50 --loss_s2sd_target_dims 512 1024 1536 2048 --loss_s2sd_feat_distill --loss_s2sd_feat_w 50 --loss_s2sd_feat_distill_delay 1000 --loss_s2sd_pool_aggr

# MDProp training for CARS196 data with Multisimilarity loss
python main.py --loss multisimilarity --dataset cars196 --source_path /path/to/data --batch_mining distance  --attacker1_targets 1 --attacker2_targets 5 --arch resnet50_normalize --bs 64 --gpu 1,2,3 --seed 3 --n_epochs 125

# MDProp training for SOP data with Multisimilarity loss
python main.py --loss multisimilarity --dataset online_products --source_path /path/to/data --batch_mining distance  --attacker1_targets 1 --attacker2_targets 5 --arch resnet50_normalize --bs 112 --gpu 1,2,3 --seed 3 --n_epochs 125

