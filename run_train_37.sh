echo 'Training...'
cd /home/wangjianbing/RLLL/
# rm -rf checkpoints/
# rm -rf logs/
# rm -rf output/
# mkdir output/
GPU_IDS=2
INIT_METHOD=tcp://127.0.0.1:43722

BATCH_SIZE=32
LOG_LEVEL=info
N_EPOCHS=30
N_EPOCHS_DECAY=10
NUM_WORKERS=0
DELS=logs_ckpts_outputs
USE_DISTRIBUTED=none

MODEL_NAME=finetune
SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "

DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
rm $DOC1
nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &

DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
rm $DOC2
nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &

tail -f $DOC1
