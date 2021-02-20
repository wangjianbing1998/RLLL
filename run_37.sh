# Train
cd /home/jianbingwang/RLLL/
rm output/ -rf
rm finetune_mnist1-mnist2-mnist3.txt
mkdir output/
nohup python train.py --n_epochs 1000 --n_epochs_decay 500 --model_name finetune --num_workers 0  --log_level info --dels logs_ckpts_outputs > finetune_mnist1-mnist2-mnist3.txt 2>&1 &
