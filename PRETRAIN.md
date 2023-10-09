# Pre-training Tubelet-Contrastive Learning

We use a single node with 4 GPUs for all pre-trainings.  We provide the **off-the-shelf** scripts in the [scripts_pretrain/](scripts_pretrain).

-  For example, to pre-train R2plus1D-18  on **Mini-Kinetcs**, you can run

  ```bash

     bash tools/dist_train.sh configs/moco/r2plus1d_18/pretraining_mini_kinetics.py  4 --data_dir /ssdstore/fmthoker/kinetics_ctp/ --work_dir ./output/moco/r2plus1d_18_mini_kinetics/pretraining 
  
  ```

-  Similarlly, to pre-train R3D-18  on **Mini-Kinetcs**, you can run

  ```bash

     bash tools/dist_train.sh configs/moco/r3d_18/pretraining_mini_kinetics.py  4 --data_dir /ssdstore/fmthoker/kinetics_ctp/ --work_dir ./output/moco/r3d_18_mini_kinetics/pretraining_100_epoch_config

  ```
-  To pre-train I3D  on **Mini-Kinetcs**, you can run

  ```bash

     bash tools/dist_train.sh configs/moco/i3d/pretraining_mini_kinetics.py  4 --data_dir /ssdstore/fmthoker/kinetics_ctp/ --work_dir ./output/moco/i3d_inetics_400/pretraining

  ```
### Note:
- We use early stopping to manually stop the training after 100 epochs.
