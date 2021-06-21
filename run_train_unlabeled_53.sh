training() {
  echo 'Training...'
  BATCH_SIZE=64
  LR=0.0001
  LOG_LEVEL=info
  N_EPOCHS=10
  N_EPOCHS_DECAY=1
  NUM_WORKERS=2
  DELS=logs_ckpts_outputs
  # DELS=none;
  LABELED_RATIO=0.2
  MAX_DATASET_SIZE=-1
}

testing() {
  echo 'Testing...'
  BATCH_SIZE=4
  LR=0.0001
  LOG_LEVEL=debug
  N_EPOCHS=1
  N_EPOCHS_DECAY=1
  NUM_WORKERS=0
  # DELS=logs_ckpts_outputs;
  DELS=none
  MAX_DATASET_SIZE=100
  LABELED_RATIO=0.2
}

training
#DATASET=mnist_cifar10_cifar100

GPU_IDS=3
sh base_run.sh lwf $LABELED_RATIO $GPU_IDS "5cifar10" $LR $N_EPOCHS $N_EPOCHS_DECAY $BATCH_SIZE $NUM_WORKERS $LOG_LEVEL $DELS $MAX_DATASET_SIZE

GPU_IDS=3
sh base_run.sh lwf $LABELED_RATIO $GPU_IDS "10cifar100" $LR $N_EPOCHS $N_EPOCHS_DECAY $BATCH_SIZE $NUM_WORKERS $LOG_LEVEL $DELS $MAX_DATASET_SIZE

#GPU_IDS=2
#
#sh base_run.sh tblwf 1.0 $GPU_IDS $PARAM
#sh base_run.sh tblwf 0.2 $GPU_IDS $PARAM

read DOC1 <DOC.txt
tail -f $DOC1
