import random
import numpy as np
import cv2
from ....datasets.transforms.dynamic_utils import (extend_key_frame_to_all, sample_key_frames)

class RandomHorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float
        The probability with which the image is flipped


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
            img_center = np.array(img.shape[:2])[::-1]/2
            img_center = np.hstack((img_center, img_center))
            if random.random() < self.p:
                img = img[:, ::-1, :]

            return img
class VerticalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float
        The probability with which the image is flipped


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self):
        pass

    def __call__(self, img,):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))

        if len(img.shape) == 3:
                img = img[::-1, :, :]
        elif len(img.shape) == 2:
                img = img[::-1, :,]

        return img


class HorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float
        The probability with which the image is flipped


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self):
        pass

    def __call__(self, img,):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))

        if len(img.shape) == 3:
                 img = img[:, ::-1, :]
        elif len(img.shape) == 2:
                 img = img[::-1,:,]

        return img

class RandomShear(object):
    """Randomly shears an image in horizontal direction   
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    shear_factor: float or tuple(float)
        if **float**, the image is sheared horizontally by a factor drawn 
        randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
        the `shear_factor` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
        
        shear_factor = random.uniform(*self.shear_factor)
        
    def __call__(self, img,):
    
        shear_factor = random.uniform(*self.shear_factor)
    
        w,h = img.shape[1], img.shape[0]
    
        if shear_factor < 0:
            img = HorizontalFlip()(img)
    
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
    
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
    
    
    
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
    
        if shear_factor < 0:
        	img = HorizontalFlip()(img)
    
        img = cv2.resize(img, (w,h))
    
    
        return img
        
class Shear(object):
    """Shears an image in horizontal direction   
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    shear_factor: float
        Factor by which the image is sheared in the x-direction
       
    Returns
    -------
    
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
    
    def __call__(self, img):
        
        shear_factor = self.shear_factor
        if shear_factor < 0:
            img = HorizontalFlip()(img,)

        
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
                
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
        

        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        
        if shear_factor < 0:
             img = HorizontalFlip()(img)
             
        
        return img
    
def shear_image_x(img,shear_factor):
        
        input_size = img.shape

        #shear along x
        if shear_factor < 0:
            img = HorizontalFlip()(img)

        #
        M = np.array([[1.0, abs(shear_factor), 0],[0,1.0,0]])
        #        
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
        #

        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
        #
        if shear_factor < 0:
             img = HorizontalFlip()(img)
        
        return img
    
def shear_image_y(img,shear_factor):
        
        input_size = img.shape
             
        #shear along y
        if shear_factor < 0:
             img = VerticalFlip()(img)

        M = np.array([[1.0, 0, 0],[abs(shear_factor),1.0,0]])

        nH =  img.shape[0] + abs(shear_factor*img.shape[1])

        img = cv2.warpAffine(img, M,  (img.shape[1],int(nH)))

        if shear_factor < 0:
             img = VerticalFlip()(img)
        
        return img
    
def shear_image_xy(img,img_alpha,shear_factor_x,shear_factor_y,shear_type):

        if shear_type==0:

             img = shear_image_x(img,shear_factor_x)
             img_alpha = shear_image_x(img_alpha,shear_factor_x)
             #print("x   shear ")

        elif shear_type==1:
             img = shear_image_y(img,shear_factor_y)
             img_alpha = shear_image_y(img_alpha,shear_factor_y)
             #print("y   shear ")

        else:
            img = shear_image_x(img,shear_factor_x)
            img_alpha = shear_image_x(img_alpha,shear_factor_x)

            img = shear_image_y(img,shear_factor_y)
            img_alpha = shear_image_y(img_alpha,shear_factor_y)
            #print("x and y  shear ")
        
        return img,img_alpha
    

def get_rotation_angles( num_frames, transform_param: dict):

        key_frame_probs = transform_param['key_frame_probs']
        loc_key_inds = sample_key_frames(num_frames, key_frame_probs)

        rot_velocity  = transform_param['rot_velocity']
        rot_angles = np.zeros((transform_param['traj_rois'].shape[0],1))

        rot_angles_list= [np.expand_dims(rot_angles, axis=0)]
        for i in range(len(loc_key_inds) - 1):
            if rot_velocity > 0:
                index_diff = loc_key_inds[i + 1] - loc_key_inds[i]
                shifts = np.random.uniform(low=-rot_velocity* index_diff,
                                           high=rot_velocity* index_diff,
                                           size=rot_angles.shape)
                rot_angles = rot_angles + shifts
            rot_angles_list.append(np.expand_dims(rot_angles, axis=0))
        rot_angles = np.concatenate(rot_angles_list, axis=0)
        rot_angles = extend_key_frame_to_all(rot_angles, loc_key_inds, 'random')
        rot_angles = rot_angles.transpose((1, 0, 2))

        return rot_angles

def get_shear_factors( num_frames, transform_param: dict):

        key_frame_probs = transform_param['key_frame_probs']
        loc_key_inds = sample_key_frames(num_frames, key_frame_probs)

        rot_velocity  = transform_param['shear_velocity']
        rot_angles = np.zeros((transform_param['traj_rois'].shape[0],1))

        #print("rotation  angles original",rot_angles.shape,loc_key_inds)
        rot_angles_list= [np.expand_dims(rot_angles, axis=0)]
        for i in range(len(loc_key_inds) - 1):
            if rot_velocity > 0:
                index_diff = loc_key_inds[i + 1] - loc_key_inds[i]
                shifts = np.random.uniform(low=-rot_velocity* index_diff,
                                           high=rot_velocity* index_diff,
                                           size=rot_angles.shape)
                rot_angles = rot_angles + shifts
            rot_angles_list.append(np.expand_dims(rot_angles, axis=0))
        rot_angles = np.concatenate(rot_angles_list, axis=0)
        rot_angles = extend_key_frame_to_all(rot_angles, loc_key_inds, 'random')
        rot_angles = rot_angles.transpose((1, 0, 2))

        return rot_angles
