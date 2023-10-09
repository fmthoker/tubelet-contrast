"""Removes missing videos from Kinetics."""
import os
import numpy as np


def load_file(file_path):
    assert os.path.isfile(file_path), f'Cannot find file: {file_path}'
    rets = []
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            if line.strip() == '':
                continue
            rets.append(line.split(' '))
    return rets


if __name__ == "__main__":
    data_dir = "/ssd/fmthoker/kinetics_ctp/"
    ann_dir = os.path.join(data_dir, "annotations")

    mini_kinetics_dir = "/ssd/fmthoker/kinetics_ctp/mini_kinetics/"

    split_files = ["train_split_1.txt", "val_split_1.txt"]
    for split_file_path in split_files:
        split_file_path_full = os.path.join(ann_dir, split_file_path)
        split_entries = load_file(split_file_path_full)
        split_ids = np.array([x[0] for x in split_entries])
        print(split_file_path_full,len(split_entries))

        split_file_path_mini = os.path.join(mini_kinetics_dir, split_file_path)
        split_entries_mini = load_file(split_file_path_mini)
        split_ids_mini = np.array([x[0] for x in split_entries_mini])
        split_labels_mini  = np.array([x[1] for x in split_entries_mini])
        print(split_file_path_mini,len(split_entries_mini))
        
        mask = np.isin(split_ids_mini, split_ids)
        print(mask.all())
        split_ids_mini = split_ids_mini[mask]
        split_labels_mini = split_labels_mini[mask]

        with open(split_file_path_mini, 'w') as f:
            for video_name, video_class in zip(split_ids_mini, split_labels_mini):
                f.write(f'{video_name} {video_class}\n')
