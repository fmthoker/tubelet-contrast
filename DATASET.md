# Data Preparation

We have successfully pre-trained our models on  [Kinetics400](https://deepmind.com/research/open-source/kinetics) with this codebase.

- The pre-processing of **Kinetics400** can be summarized into following steps:

  1. Download the dataset from [official website](https://deepmind.com/research/open-source/kinetics).

  2. Assuming the following file structure:  
     ```
     dataset_root/VideoData/abseiling/*.avi
     dataset_root/VideoData/air_drumming/*.avi
     ...
     dataset_root/VideoData/zumba/*.avi
     ```
     dataset_root/labels/kinetics_train.csv
     dataset_root/labels/kinetics_val.csv
     dataset_root/labels/classInd.txt
     dataset_root/labels/missing_train_videofolder.txt
     dataset_root/labels/missing_val_videofolder.txt

     We share our these files in **[kinetics labels](./data/labels/)**.

  3. Run the follwing script to zip video files 

     ```
     # set the correct dataset path and desired output path
     python scripts_preprocess/process_kinetics.py

     ```
  4. Output Structure 

     ```
     # set of zip files for all kinetics-400 videos
     output_path/kinetics_processed/*.zip

     ```
  4. Generate Text annotations into JSON  for mini-kinetics and kinetics-400 datasets

     ```
     # python  cvt_txt_to_json.py 

     ```
     We share our annotation files for mini-kinetcs and kinetics-400 in **[kinetics labels](./data/annotations/)**.

