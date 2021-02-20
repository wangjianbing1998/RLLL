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

