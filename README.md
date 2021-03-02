# RLLL
R-LifeLongLearning


# Model
- finetune: frozen other layers, unuse the other loss
- warmtune: frozen the shared-cnn layers
- hottune: unfrozen the shared-cnn layers
- folwf: frozen other layers' LWF
- lwf: learning without forgetting, unfrozen all layers and use the other loss
- rlll(Ours): unfrozen all layers and use the other loss, use the backward training



## Table correlation
- Q-False
- R-True

|          | shared_cnn_layers | shared_fc_layers | new_layers | other_layers | other_loss | Backward |
|----------|-------------------|------------------|------------|--------------|------------|----------|
| finetune | Q                 | Q                | R          | Q            | Q          | Q        |
| warmtune | Q                 | R                | R          | Q            | Q          | Q        |
| hottune  | R                 | R                | R          | Q            | Q          | Q        |
| fo-lwf   | R                 | R                | R          | Q            | R          | Q        |
| lwf      | R                 | R                | R          | R            | R          | Q        |
| RLLL(Ours)     | R                 | R                | R          | R            | R          | R        |



# DataPrepare
```sh
cd data
```

Only need to prepare MNIST data or none, because data will be prepare in train.py or test.py with `--load_dataset_mode=reader`, here, `--xxx-dataset_dir` need to be specified,

For Examples:

```sh
python train.py --load_dataset_mode reader --cifar100-dataset-dir ./data/CIFAR100
```

- prepare MNIST data

```python
python mnist_download.py
```

# Train
- run at #66 server
```sh
sh run_66.sh
```


- run at #53 server
```sh
sh run_53.sh
```


- run at #37 server
```shell
sh run_37.sh
```



# Test
```sh
python test.py
```