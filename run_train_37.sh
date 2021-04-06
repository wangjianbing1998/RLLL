echo 'Training...'
cd /home/jianbingwang/RLLL/
# rm -rf checkpoints/
# rm -rf logs/
# rm -rf output/
# mkdir output/


INIT_METHOD=tcp://127.0.0.1:43722
BATCH_SIZE=64
LOG_LEVEL=info
N_EPOCHS=90
N_EPOCHS_DECAY=20
NUM_WORKERS=0
DELS=logs_ckpts_outputs
USE_DISTRIBUTED=none
MAX_DATASET_SIZE=9

GPU_IDS=1
SUFFIX=" --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS --max_dataset_size $MAX_DATASET_SIZE"


MODEL_NAME=finetune
DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
rm $DOC1
nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &


MODEL_NAME=lwf
DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
rm $DOC1
nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &



MODEL_NAME=tblwf
DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
rm $DOC1
nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &


MODEL_NAME=halwf
DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &


MODEL_NAME=walwf
DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &



MODEL_NAME=falwf
DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
rm $DOC1
nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &


# MODEL_NAME=finetune
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &



# MODEL_NAME=lwf
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &



# MODEL_NAME=tblwf
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &



#
# MODEL_NAME=halwf
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &
#


# MODEL_NAME=lwf
# SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
# DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
# rm $DOC1
# nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &
# tail -f $DOC1



# MODEL_NAME=falwf
# SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
# DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
# rm $DOC1
# nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &
# tail -f $DOC1
#
#
# GPU_IDS=1

# DOC1=$MODEL_NAME"_train_mnist_1-mnist_2-mnist_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list mnist_1 mnist_2 mnist_3 $SUFFIX --model_name $MODEL_NAME > $DOC1 2>&1 &

#
# DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# rm $DOC2
# nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &
#
# #
# MODEL_NAME=warmtune
# SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
# # DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
# # rm $DOC1
# # nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &
#
# # DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# # rm $DOC2
# # nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &
#
#
#
#
# MODEL_NAME=hottune
# SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
# DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
# rm $DOC1
# nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &
# #
# # DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# # rm $DOC2
# # nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &
#
#
# GPU_IDS=2
# MODEL_NAME=folwf
# SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
# DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
# rm $DOC1
# nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &
#
# # # DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# # # rm $DOC2
# # # nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &
# #
# #
# MODEL_NAME=lwf
# SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
# DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
# rm $DOC1
# nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &
#
# # DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# # rm $DOC2
# # nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &
#
#
#
# MODEL_NAME=tblwf
# SUFFIX=" --model_name $MODEL_NAME --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --gpu_ids $GPU_IDS "
# # # DOC1=$MODEL_NAME"_train_mnist-cifar10-cifar100.txt"
# # # rm $DOC1
# # # nohup python train.py --dataset_list mnist cifar10 cifar100 $SUFFIX > $DOC1 2>&1 &
#
# DOC2=$MODEL_NAME"_train_cifar100-cifar10-mnist.txt"
# rm $DOC2
# nohup python train.py --dataset_list cifar100 cifar10 mnist $SUFFIX > $DOC2 2>&1 &
#
tail -f $DOC1
