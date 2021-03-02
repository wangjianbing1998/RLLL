echo 'Training...'
cd /home/wangjianbing/RLLL/
mkdir output/

rm rlll_mnist1-mnist2-mnist3.txt
# nohup python train.py --n_epochs 50 --n_epochs_decay 30 --batch_size 8 --model_name rlll --num_workers 0  --log_level info --dels logs_ckpts_outputs --init_method tcp://127.0.0.1:46622 > rlll_mnist1-mnist2-mnist3.txt 2>&1 &

rm rlll_mnist1-mnist2-cifar10.txt
# python train.py --model_name rlll --dataset_list mnist_1 mnist_2 cifar10 --n_epochs 50 --n_epochs_decay 30 --batch_size 8 --num_workers 0  --log_level info --dels logs_ckpts_outputs --init_method tcp://127.0.0.1:46622 --log_filename None

rm rlll_cifar100-cifar10-mnist.txt
nohup python train.py --model_name rlll --dataset_list cifar100 cifar10 mnist --n_epochs 50 --n_epochs_decay 30 --batch_size 8 --num_workers 0  --log_level info --dels logs_ckpts_outputs --init_method tcp://127.0.0.1:46622 > rlll_cifar100-cifar10-mnist.txt 2>&1 &
# python train.py --model_name rlll --dataset_list mnist cifar10 cifar100 --n_epochs 50 --n_epochs_decay 30 --batch_size 8 --num_workers 0  --log_level info --dels logs_ckpts_outputs --init_method tcp://127.0.0.1:46622
