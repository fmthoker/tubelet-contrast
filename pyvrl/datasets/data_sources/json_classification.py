import os
import json
import random
import numpy as np
import math

def get_subset_data(video_names,annotations,seed,data_percentage):

    print("data lengths before subsampling",len(video_names),len(annotations))

    random_state = np.random.RandomState(seed)

    annotations = np.array(annotations)
    video_names = np.array(video_names)

    subset_video_names = []
    subset_annotations = []
    total_examples = 0 
    for class_label in set(annotations):

          subset_indexes = np.where(annotations == class_label)
          #print("examples of class",class_label,subset_indexes[0].shape[0])
          total_examples += subset_indexes[0].shape[0]

          samples_class = subset_indexes[0].shape[0]
          ran_indicies = np.array(random_state.choice(samples_class,int(math.ceil(samples_class * data_percentage)), replace=False))

          indicies_100 = (subset_indexes[0][ran_indicies])
          temp_annotations =annotations[indicies_100]
          temp_names=  video_names[indicies_100]
          subset_video_names.extend(temp_names)
          subset_annotations.extend(temp_annotations)
    #print("total examples of all classes",total_examples)
    video_names  = list(subset_video_names)
    annotations = list(subset_annotations)
    print("data lengths after subsampling",len(video_names),len(annotations))
    #print(video_names)
    #print(annotations)
    return video_names, annotations


class JsonClsDataSource(object):

    def __init__(self, ann_file: str, data_dir: str = None):
        """ The video name & class label are stored in a json file. """
        self.data_dir = data_dir
        if data_dir is not None:
            ann_file = os.path.join(data_dir, ann_file)
        self.ann_file = ann_file
        assert self.ann_file.endswith('.json'), f'Support .json file only, but got {ann_file}'
        assert os.path.isfile(self.ann_file), f'Cannot find file {ann_file}'
        with open(self.ann_file, 'r') as f:
            self.video_info_list = json.load(f)


       
        video_names = []
        annotations = []
        for item in  self.video_info_list:
                 video_names.append(item['name'])
                 annotations.append(item['label'])

        #seed = 12345
        seed = 99999
        seed = 4567980
        #print("video list",self.video_info_list[0:100])
        #data_percentage = 0.05
        #data_percentage = 0.10 
        #data_percentage = 0.25
        #data_percentage = 0.33
        #data_percentage = 0.50
        #data_percentage = 1.0
        #video_names, annotations = get_subset_data(video_names,annotations,seed,data_percentage)
        #video_info_list_new = []
        #for n,l in zip(video_names,annotations):
        #       item =  {'name':n , 'label': l}
        #       video_info_list_new.append(item)

        #self.video_info_list = video_info_list_new
        #print("video list new ",self.video_info_list[0:100])


    def __len__(self):
        return len(self.video_info_list)

    def __getitem__(self, idx):
        return self.video_info_list[idx]
