# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Data preprocessing for Kinetics dataset. """
import os
import zipfile
import mmcv
import cv2
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess Kinetics dataset')
    parser.add_argument('--raw_dir', default='/ssd/fmthoker/kinetics/VideoData/',
                        type=str, help='raw data directory')
    parser.add_argument('--out_dir', default='/ssd/fmthoker/kinetics_ctp/',
                        type=str, help='output data directory.')
    parser.add_argument('--ann_dir', default='/ssd/fmthoker/kinetics/labels/',
                        type=str, help='train/test split annotations directory.')
    return parser.parse_args()


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


def video_to_zip(video_path, zip_path):
    assert os.path.isfile(video_path)
    vid = cv2.VideoCapture(video_path)
    mmcv.mkdir_or_exist(os.path.dirname(zip_path))
    zid = zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED)
    frame_index = 1
    while True:
        ret, img = vid.read()
        if img is None:
            break
        img_str = cv2.imencode('.jpg', img)[1].tostring()
        zid.writestr('img_{:05d}.jpg'.format(frame_index), img_str)
        frame_index += 1
    zid.close()
    vid.release()


if __name__ == '__main__':
    args = parse_args()

    # Step 1, load class ID annotations
    ids_file = os.path.join(args.ann_dir, 'classInd.txt')
    if not os.path.isfile(ids_file):
        classes = os.listdir(os.path.join(args.raw_dir))
        classes = sorted(classes)
        ids_list = [f"{idx+1} {_class}" for idx, _class in enumerate(classes)]
        with open(ids_file, 'w') as f:
            for item in ids_list:
                f.write("%s\n" % item)
    ids_map = {sp[1]: int(sp[0]) for sp in load_file(ids_file)}

    # Step 2, load training & test annotations
    all_video_name_list = []
    all_video_label_list = []
    for prefix in ['train', 'val']:
        ann_file = pd.read_csv(os.path.join(args.ann_dir, f"kinetics_{prefix}.csv"))
        video_id_list = list(ann_file.youtube_id.values)
        ann_file.time_start = ann_file.time_start.astype(str).apply(lambda x: x.zfill(6))
        ann_file.time_end = ann_file.time_end.astype(str).apply(lambda x: x.zfill(6))
        ann_file["video_name"] = ann_file[["youtube_id", "time_start", "time_end"]].apply(
            lambda x: "_".join(x), axis=1
        )

        # remove missing videos
        missing = load_file(os.path.join(args.ann_dir, f"missing_{prefix}_videofolder.txt"))
        missing = [os.path.basename(x[0]) for x in missing]
        #print(len(missing))
        ann_file = ann_file[~ann_file.video_name.isin(missing)].reset_index()

        video_name_list = list(ann_file.video_name.values)
        ann_file.label = ann_file.label.apply(lambda x: x.replace(" ", "_").replace("(", "").replace(")","").replace("'", ""))
        video_label_list = list(ann_file.label.values)
        video_class_list = [ids_map[x] for x in video_label_list]

        out_file = os.path.join(args.out_dir, 'annotations', f'{prefix}_split_1.txt')
        mmcv.mkdir_or_exist(os.path.dirname(out_file))

        with open(out_file, 'w') as f:
            for video_name, video_class in zip(video_name_list, video_class_list):
                f.write(f'{video_name} {video_class}\n')
        all_video_name_list.extend(video_name_list)
        all_video_label_list.extend(video_label_list)

    # Step 3, convert .avi raw video to zipfile
    prog_bar = mmcv.ProgressBar(len(all_video_name_list))
    missing_videos_not_listed = []
    for video_label, video_name in zip(all_video_label_list, all_video_name_list):
        video_path = os.path.join(args.raw_dir, video_label, video_name + ".avi")
        zip_path = os.path.join(args.out_dir, 'zips', f'{video_name}.zip')

        if not os.path.isfile(video_path):
            print(f"Video does not exist at {video_path}")
            missing_videos_not_listed.append(video_path)
            continue

        if not os.path.exists(zip_path):
            video_to_zip(video_path, zip_path)

        prog_bar.update()

    out_file = os.path.join(args.out_dir, 'annotations', f'missing_unlisted_train_val.txt')
    with open(out_file, 'w') as f:
        for video_path in missing_videos_not_listed:
            f.write(f'{video_path}\n')

