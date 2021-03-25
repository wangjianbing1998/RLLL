echo 'Training...'
cd /home/wangjianbing/RLLL/
# rm -rf checkpoints/
# rm -rf logs/
# rm -rf output/
# mkdir output/


INIT_METHOD=tcp://127.0.0.1:45322
BATCH_SIZE=64
LOG_LEVEL=info
N_EPOCHS=90
N_EPOCHS_DECAY=20
NUM_WORKERS=0
DELS=logs_ckpts_outputs
USE_DISTRIBUTED=none


GPU_IDS=1
MODEL_NAME=finetune
SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
# DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
# rm $DOC1
# nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &

DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
rm $DOC2
nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &


MODEL_NAME=warmtune
SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
rm $DOC1
nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &

# DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# rm $DOC2
# nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &




MODEL_NAME=hottune
SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
rm $DOC1
nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &

DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
rm $DOC2
nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &



GPU_IDS=2
MODEL_NAME=folwf
SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
rm $DOC1
nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &

# DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# rm $DOC2
# nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &


MODEL_NAME=lwf
SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
rm $DOC1
nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &

DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
rm $DOC2
nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &



MODEL_NAME=tblwf
SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
rm $DOC1
nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &

# DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# rm $DOC2
# nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &

tail -f $DOC2
