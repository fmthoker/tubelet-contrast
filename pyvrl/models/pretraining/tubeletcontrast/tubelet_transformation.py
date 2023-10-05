
import math
import numpy as np
import random
import cv2
import imutils
from .utils import get_rotation_angles, get_shear_factors
from .utils import shear_image_xy

def tubelet_rotation(data_1,data_2,transform_param,arg_rank):

         #print(" patch transformation=",transform_param['patch_transformation'])
         # get rotation angles
         rot_angles = get_rotation_angles(len(data_1),transform_param)
         width, height,_ = data_1[0].shape
         size = width

         transformed_data_1 = []
         for frame_idx in range(len(data_1)):
             i_rois = transform_param['traj_rois'][:, frame_idx, :]
             img = data_1[frame_idx].copy()
             for patch_idx in arg_rank:
                 if not transform_param['traj_labels'][patch_idx][frame_idx]:
                     continue
                 i_patch = transform_param['patches'][patch_idx][frame_idx]
                 i_alpha = transform_param['alphas'][patch_idx][frame_idx]

                 angle = int(rot_angles[patch_idx][frame_idx])
                 rotated_i_patch = imutils.rotate_bound(i_patch, angle)
                 rotated_i_alpha = imutils.rotate_bound(i_alpha, angle)

                 h_prime,w_prime,channels = rotated_i_patch.shape
                 x1, y1, x2, y2 = i_rois[patch_idx]
                 h, w  = y2-y1, x2-x1
                 if ((h_prime -h) % 2) == 0: 
                      delta_h1 = delta_h2 = math.ceil((h_prime -h)/2)
                 else:
                      delta_h1 = math.ceil((h_prime -h)/2)
                      delta_h2 = math.floor((h_prime -h)/2)
                 if ((w_prime -w) % 2) == 0: 
                      delta_w1 = delta_w2 = math.ceil((w_prime -w)/2)
                 else:
                      delta_w1 = math.ceil((w_prime -w)/2)
                      delta_w2 = math.floor((w_prime -w)/2)

                 x1_new, y1_new, x2_new, y2_new =  x1-delta_w1, y1-delta_h1, x2+delta_w2, y2+delta_h2
                 if all(i >= 0 for i in [x1_new, y1_new, x2_new, y2_new]) and all(i < size for i in [x1_new, y1_new, x2_new, y2_new]):
                        #print("in bound")
                        i_alpha = rotated_i_alpha[..., np.newaxis]
                        i_patch = rotated_i_patch

                        img[y1_new:y2_new, x1_new:x2_new, :] = img[y1_new:y2_new, x1_new:x2_new, :] * (1 - i_alpha) + \
                                        i_patch * i_alpha
                 else:
                        #print("out of bound")
                        temp = np.array((x1_new, y1_new, x2_new, y2_new))
                        temp = np.clip(temp ,0,size )

                        x1_new, y1_new, x2_new, y2_new =temp
                        row,col,channel = np.indices(img[y1_new:y2_new, x1_new:x2_new, :].shape)
                        i_patch = rotated_i_patch[row,col,channel]

                        i_alpha = rotated_i_alpha[..., np.newaxis]
                        row,col = np.indices(img[y1_new:y2_new, x1_new:x2_new,0].shape)
                        i_alpha = i_alpha[row,col]

                        img[y1_new:y2_new, x1_new:x2_new, :] = img[y1_new:y2_new, x1_new:x2_new, :] * (1 - i_alpha) + \
                                        i_patch * i_alpha

             transformed_data_1.append(img)

         transformed_data_2 = []


         for frame_idx in range(len(data_2)):
             i_rois = transform_param['traj_rois'][:, frame_idx, :]
             img = data_2[frame_idx].copy()
             for patch_idx in arg_rank:
                 if not transform_param['traj_labels_2'][patch_idx][frame_idx]:
                     continue
                 i_patch = transform_param['patches_2'][patch_idx][frame_idx]
                 i_alpha = transform_param['alphas_2'][patch_idx][frame_idx]

                 angle = int(rot_angles[patch_idx][frame_idx])
                 rotated_i_patch = imutils.rotate_bound(i_patch, angle)
                 rotated_i_alpha = imutils.rotate_bound(i_alpha, angle)

                 h_prime,w_prime,channels = rotated_i_patch.shape

                 x1, y1, x2, y2 = i_rois[patch_idx]
                 h, w  = y2-y1, x2-x1
                 if ((h_prime -h) % 2) == 0: 
                      delta_h1 = delta_h2 = math.ceil((h_prime -h)/2)
                 else:
                      delta_h1 = math.ceil((h_prime -h)/2)
                      delta_h2 = math.floor((h_prime -h)/2)
                 if ((w_prime -w) % 2) == 0: 
                      delta_w1 = delta_w2 = math.ceil((w_prime -w)/2)
                 else:
                      delta_w1 = math.ceil((w_prime -w)/2)
                      delta_w2 = math.floor((w_prime -w)/2)

                 x1_new, y1_new, x2_new, y2_new =  x1-delta_w1, y1-delta_h1, x2+delta_w2, y2+delta_h2
                 if all(i >= 0 for i in [x1_new, y1_new, x2_new, y2_new]) and all(i < size for i in [x1_new, y1_new, x2_new, y2_new]):
                        i_alpha = rotated_i_alpha[..., np.newaxis]
                        i_patch = rotated_i_patch


                        img[y1_new:y2_new, x1_new:x2_new, :] = img[y1_new:y2_new, x1_new:x2_new, :] * (1 - i_alpha) + \
                                        i_patch * i_alpha
                 else:
                        temp = np.array((x1_new, y1_new, x2_new, y2_new))
                        temp = np.clip(temp ,0,size )

                        x1_new, y1_new, x2_new, y2_new =temp
                        row,col,channel = np.indices(img[y1_new:y2_new, x1_new:x2_new, :].shape)
                        i_patch = rotated_i_patch[row,col,channel]

                        i_alpha = rotated_i_alpha[..., np.newaxis]
                        row,col = np.indices(img[y1_new:y2_new, x1_new:x2_new,0].shape)
                        i_alpha = i_alpha[row,col]

                        img[y1_new:y2_new, x1_new:x2_new, :] = img[y1_new:y2_new, x1_new:x2_new, :] * (1 - i_alpha) + \
                                        i_patch * i_alpha

             transformed_data_2.append(img)


         transformed_data =  transformed_data_1 + transformed_data_2
         return transformed_data

def tubelet_shearing(data_1,data_2,transform_param,arg_rank):
         #print(" patch transformation=",transform_param['patch_transformation'])

         width, height,_ = data_1[0].shape
         size = width

         shear_angles_x = get_shear_factors(len(data_1),transform_param)
         shear_angles_y = get_shear_factors(len(data_1),transform_param)

         shear_type =np.random.choice(3)


         transformed_data_1 = []
         for frame_idx in range(len(data_1)):
             i_rois = transform_param['traj_rois'][:, frame_idx, :]
             img = data_1[frame_idx].copy()
             for patch_idx in arg_rank:
                 if not transform_param['traj_labels'][patch_idx][frame_idx]:
                     continue
                 i_patch = transform_param['patches'][patch_idx][frame_idx]
                 i_alpha = transform_param['alphas'][patch_idx][frame_idx]

                 angle_x = shear_angles_x[patch_idx][frame_idx].item()
                 angle_y = shear_angles_y[patch_idx][frame_idx].item()
                 shear_i_patch,shear_i_alpha = shear_image_xy(i_patch,i_alpha, angle_x,angle_y,shear_type)


                 h_prime,w_prime,channels = shear_i_patch.shape
                 x1, y1, x2, y2 = i_rois[patch_idx]
                 h, w  = y2-y1, x2-x1
                 if ((h_prime -h) % 2) == 0: 
                      delta_h1 = delta_h2 = math.ceil((h_prime -h)/2)
                 else:
                      delta_h1 = math.ceil((h_prime -h)/2)
                      delta_h2 = math.floor((h_prime -h)/2)
                 if ((w_prime -w) % 2) == 0: 
                      delta_w1 = delta_w2 = math.ceil((w_prime -w)/2)
                 else:
                      delta_w1 = math.ceil((w_prime -w)/2)
                      delta_w2 = math.floor((w_prime -w)/2)

                 x1_new, y1_new, x2_new, y2_new =  x1-delta_w1, y1-delta_h1, x2+delta_w2, y2+delta_h2
                 if all(i >= 0 for i in [x1_new, y1_new, x2_new, y2_new]) and all(i < width for i in [x1_new, y1_new, x2_new, y2_new]):
                        #print("in bound")
                        i_alpha = shear_i_alpha[..., np.newaxis]
                        i_patch = shear_i_patch

                        img[y1_new:y2_new, x1_new:x2_new, :] = img[y1_new:y2_new, x1_new:x2_new, :] * (1 - i_alpha) + \
                                        i_patch * i_alpha
                 else:
                        #print("out of bound")
                        temp = np.array((x1_new, y1_new, x2_new, y2_new))
                        temp = np.clip(temp ,0,width )

                        x1_new, y1_new, x2_new, y2_new =temp
                        row,col,channel = np.indices(img[y1_new:y2_new, x1_new:x2_new, :].shape)
                        i_patch = shear_i_patch[row,col,channel]

                        i_alpha = shear_i_alpha[..., np.newaxis]
                        row,col = np.indices(img[y1_new:y2_new, x1_new:x2_new,0].shape)
                        i_alpha = i_alpha[row,col]

                        img[y1_new:y2_new, x1_new:x2_new, :] = img[y1_new:y2_new, x1_new:x2_new, :] * (1 - i_alpha) + \
                                        i_patch * i_alpha

             transformed_data_1.append(img)

         transformed_data_2 = []


         for frame_idx in range(len(data_2)):
             i_rois = transform_param['traj_rois'][:, frame_idx, :]
             img = data_2[frame_idx].copy()
             for patch_idx in arg_rank:
                 if not transform_param['traj_labels_2'][patch_idx][frame_idx]:
                     continue
                 i_patch = transform_param['patches_2'][patch_idx][frame_idx]
                 i_alpha = transform_param['alphas_2'][patch_idx][frame_idx]

                 angle_x = shear_angles_x[patch_idx][frame_idx].item()
                 angle_y = shear_angles_y[patch_idx][frame_idx].item()
                 shear_i_patch,shear_i_alpha = shear_image_xy(i_patch,i_alpha, angle_x,angle_y,shear_type)

                 h_prime,w_prime,channels = shear_i_patch.shape

                 x1, y1, x2, y2 = i_rois[patch_idx]
                 h, w  = y2-y1, x2-x1
                 if ((h_prime -h) % 2) == 0: 
                      delta_h1 = delta_h2 = math.ceil((h_prime -h)/2)
                 else:
                      delta_h1 = math.ceil((h_prime -h)/2)
                      delta_h2 = math.floor((h_prime -h)/2)
                 if ((w_prime -w) % 2) == 0: 
                      delta_w1 = delta_w2 = math.ceil((w_prime -w)/2)
                 else:
                      delta_w1 = math.ceil((w_prime -w)/2)
                      delta_w2 = math.floor((w_prime -w)/2)

                 x1_new, y1_new, x2_new, y2_new =  x1-delta_w1, y1-delta_h1, x2+delta_w2, y2+delta_h2
                 if all(i >= 0 for i in [x1_new, y1_new, x2_new, y2_new]) and all(i < width for i in [x1_new, y1_new, x2_new, y2_new]):
                        #print("in bound")
                        i_alpha = shear_i_alpha[..., np.newaxis]
                        i_patch = shear_i_patch

                        img[y1_new:y2_new, x1_new:x2_new, :] = img[y1_new:y2_new, x1_new:x2_new, :] * (1 - i_alpha) + \
                                        i_patch * i_alpha
                 else:
                        temp = np.array((x1_new, y1_new, x2_new, y2_new))
                        temp = np.clip(temp ,0, width )

                        x1_new, y1_new, x2_new, y2_new =temp
                        row,col,channel = np.indices(img[y1_new:y2_new, x1_new:x2_new, :].shape)
                        i_patch = shear_i_patch[row,col,channel]

                        i_alpha = shear_i_alpha[..., np.newaxis]
                        row,col = np.indices(img[y1_new:y2_new, x1_new:x2_new,0].shape)
                        i_alpha = i_alpha[row,col]

                        img[y1_new:y2_new, x1_new:x2_new, :] = img[y1_new:y2_new, x1_new:x2_new, :] * (1 - i_alpha) + \
                                        i_patch * i_alpha

             transformed_data_2.append(img)


         transformed_data =  transformed_data_1 + transformed_data_2
         return transformed_data
