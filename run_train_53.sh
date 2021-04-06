echo 'Training...'
# rm -rf checkpoints/
# rm -rf logs/
# rm -rf output/
# mkdir output/


INIT_METHOD=tcp://127.0.0.1:45322
BATCH_SIZE=1024
LOG_LEVEL=info
N_EPOCHS=100
N_EPOCHS_DECAY=1
NUM_WORKERS=4
# DELS=logs_ckpts_outputs
DELS=none
USE_DISTRIBUTED=none
MAX_DATASET_SIZE=-1


SUFFIX=" --continue_train --n_epochs $N_EPOCHS --use_distributed $USE_DISTRIBUTED --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --init_method $INIT_METHOD --max_dataset_size $MAX_DATASET_SIZE"

# GPU_IDS=1
#
#
# MODEL_NAME=finetune
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS --gpu_ids $GPU_IDS > $DOC1 2>&1 &
# #
# DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &

# GPU_IDS=2
#
# MODEL_NAME=lwf
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &


# GPU_IDS=2
#
# MODEL_NAME=tblwf
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &




# MODEL_NAME=halwf
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &




#
# DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &

GPU_IDS=2


# MODEL_NAME=walwf
# DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
# #
# DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#


MODEL_NAME=falwf
DOC1=$MODEL_NAME"_train_miniimagenet_1-miniimagenet_2-miniimagenet_3.txt"
rm $DOC1
nohup python train.py --dataset_list miniimagenet_1 miniimagenet_2 miniimagenet_3 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_miniimagenet-cifar100-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list miniimagenet cifar100 cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar100-miniimagenet-cifar10.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar100 miniimagenet cifar10 $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &
#
# DOC1=$MODEL_NAME"_train_cifar10-cifar100-miniimagenet.txt"
# rm $DOC1
# nohup python train.py --dataset_list cifar10 cifar100 miniimagenet $SUFFIX --model_name $MODEL_NAME --gpu_ids $GPU_IDS > $DOC1 2>&1 &


tail -f $DOC1