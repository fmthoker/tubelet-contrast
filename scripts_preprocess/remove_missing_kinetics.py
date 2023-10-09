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

    missing_videos = load_file(os.path.join(ann_dir, "missing_unlisted_train_val.txt"))
    missing_videos = [os.path.basename(x[0]).split(".avi")[0] for x in missing_videos]
    missing_videos = np.array(missing_videos)

    split_files = ["train_split_1.txt", "val_split_1.txt"]
    for split_file_path in split_files:
        split_file_path = os.path.join(ann_dir, split_file_path)
        split_entries = load_file(split_file_path)
        split_ids = np.array([x[0] for x in split_entries])
        split_labels = np.array([x[1] for x in split_entries])
        
        mask = np.isin(split_ids, missing_videos)
        split_ids = split_ids[np.bitwise_not(mask)]
        split_labels = split_labels[np.bitwise_not(mask)]

        with open(split_file_path, 'w') as f:
            for video_name, video_class in zip(split_ids, split_labels):
                f.write(f'{video_name} {video_class}\n')
