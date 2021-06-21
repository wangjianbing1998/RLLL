#base_run.sh

if [ $# -ne 12 ]; then
  echo "Usage is $0 MODEL_NAME LABELED_RATIO GPU_IDS DATASET, But got "$#
  exit
fi

MODEL_NAME=$1
LABELED_RATIO=$2
GPU_IDS=$3
DATASET=$4
LR=$5
N_EPOCHS=$6
N_EPOCHS_DECAY=$7
BATCH_SIZE=$8
NUM_WORKERS=$9
LOG_LEVEL=${10}
DELS=${11}
MAX_DATASET_SIZE=${12}

SUFFIX=" --labeled_ratio $LABELED_RATIO --lr $LR --n_epochs $N_EPOCHS --continue_train --n_epochs_decay $N_EPOCHS_DECAY --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS  --log_level $LOG_LEVEL --dels $DELS --max_dataset_size $MAX_DATASET_SIZE"
#SPLIT_DATASET=`echo $DATASET|tr '_' ' '`;

if [ $(echo "$LABELED_RATIO == 1" | bc) -eq 0 ]; then

  DOC1=$LABELED_RATIO"_"$MODEL_NAME"_"$DATASET"_train.txt"
  echo "LABELED_RATIO != 1.0, then unuse unlabeled data, saved into "$DOC1
  CUDA_VISIBLE_DEVICES=$GPU_IDS
  nohup python3 train.py --dataset_list $DATASET --model_name $MODEL_NAME --gpu_ids $GPU_IDS $SUFFIX >$DOC1 2>&1 &

  DOC1=$LABELED_RATIO"_"$MODEL_NAME"_"$DATASET"_unlabeled_train.txt"
  echo "LABELED_RATIO != 1.0, then use unlabeled data, saved into "$DOC1
  CUDA_VISIBLE_DEVICES=$GPU_IDS
  nohup python3 train.py --unlabeled --dataset_list $DATASET --model_name $MODEL_NAME --gpu_ids $GPU_IDS $SUFFIX >$DOC1 2>&1 &

else

  DOC1=$LABELED_RATIO"_"$MODEL_NAME"_"$DATASET"_train.txt"
  echo "LABELED_RATIO == 1.0, then use all labeled data, saved into "$DOC1
  CUDA_VISIBLE_DEVICES=$GPU_IDS
  nohup python3 train.py --dataset_list $DATASET --model_name $MODEL_NAME --gpu_ids $GPU_IDS $SUFFIX >$DOC1 2>&1 &
fi

echo $DOC1 >DOC.txt
