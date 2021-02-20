# Train

nohup python train.py --n_epochs 1000 --n_epochs_decay 500 --model_name lwf --num_workers 0  --log_level info --dels logs_ckpts_outputs &
nohup python train.py --n_epochs 1000 --n_epochs_decay 500 --model_name finetune --num_workers 0 --log_level info --dels logs_ckpts_outputs &
nohup python train.py --n_epochs 1000 --n_epochs_decay 500 --model_name warmtune --num_workers 0 --log_level info --dels logs_ckpts_outputs &
nohup python train.py --n_epochs 1000 --n_epochs_decay 500 --model_name hottune --num_workers 0 --log_level info --dels logs_ckpts_outputs &
nohup python train.py --n_epochs 1000 --n_epochs_decay 500 --model_name folwf --num_workers 0 --log_level info --dels logs_ckpts_outputs &
nohup python train.py --n_epochs 1000 --n_epochs_decay 500 --model_name rlll --num_workers 0 --log_level info --dels logs_ckpts_outputs &
