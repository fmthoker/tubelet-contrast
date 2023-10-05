from torch.utils.data import Dataset


from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder
import random


@DATASETS.register_module()
class MoCoDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 frame_sampler: dict,
                 transform_cfg: list,
                 transform_cfg_2: list,
                 test_mode: bool = False):
        """ A dataset class to generate a pair of training examples
        for contrastive learning. Basically, the vanilla MoCo is traine on
        image dataset, like ImageNet-1M. To facilitate its pplication on video
        dataset, we random pick two video clip and discriminate whether
        these two clips are from same video or not.

        Args:
            data_source (dict): data source configuration dictionary
            data_dir (str): data root directory
            transform_cfg (list): data augmentation configuration list
            backend (dict): storage backend configuration
            test_mode (bool): placeholder, not available in MoCo training.
        """
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source, dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.frame_sampler = builder.build_frame_sampler(frame_sampler)
        self.img_transform = Compose(transform_cfg)
        self.img_transform_2 = Compose(transform_cfg_2)

        self.test_mode = test_mode

    def __len__(self):
        return len(self.data_source)

    def get_single_clip(self, video_info, storage_obj):
        """ Get single video clip according to the video_info query."""
        frame_inds = self.frame_sampler.sample(len(storage_obj))
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1
        img_list = storage_obj.get_frame(frame_inds.reshape(-1))
        return img_list


    def get_tubelet_based_clips(self,idx):

        try:
               video_info = self.data_source[idx]
               # build video storage backend object
               storage_obj = self.backend.open(video_info)
               img_list_1 = self.get_single_clip(video_info,storage_obj)
               img_list_1, _ = self.img_transform.apply_image(img_list_1,
                                           return_transform_param=True)
        except Exception:
               return self[(idx+1) % len(self)]
        try:
               idx_2 = random.randint(0, len(self))
               #print("idxs",idx,idx_2)
               video_info_2 = self.data_source[idx_2]
               # build video storage backend object
               storage_obj_2 = self.backend.open(video_info_2)
               img_list_2 = self.get_single_clip(video_info_2,storage_obj_2)
               img_list_2, _ = self.img_transform.apply_image(img_list_2,
                                           return_transform_param=True)
        except Exception:
               return self[(idx+1) % len(self)]

        img_list = img_list_1 + img_list_2

        img_tensor, trans_params = \
            self.img_transform_2.apply_image(img_list,
                                           return_transform_param=True)

        clip_len = int(img_tensor.size(0) / 2)
        img_tensor_1 = img_tensor[0:clip_len,:,:,:].permute(1, 0, 2, 3).contiguous()
        img_tensor_2 = img_tensor[clip_len:,:,:,:].permute(1, 0, 2, 3).contiguous()
        #print(img_tensor_1.size(),img_tensor_2.size())

        data = dict(
            imgs=img_tensor_1,
            imgs_k=img_tensor_2
        )
        storage_obj.close()
        storage_obj_2.close()

        return data

    def __getitem__(self, idx):

        data  = self.get_tubelet_based_clips(idx)

        return data

