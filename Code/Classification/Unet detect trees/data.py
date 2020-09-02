from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

Tree = [255,255,255]
Background = [0,0,0]

COLOR_DICT = np.array([Tree, Background])

THRESHOLD = 255/2 # all colors which lay within this threshold are of this class; This is because of slight changes within the image, which can happen during augmentation or loading; Take care for this when creating the mask


def adjustData(img,mask,flag_multi_class, num_class, true_value = 1): # num_class is obsolete, since in this version of the method it isn't used
    wrong_value = 1-true_value # usually this is 0..
    wrong_value_rate = wrong_value / (len(COLOR_DICT)-1) # the value of each pixel should sum to 1: e.g. true value: 1, wrong_value_1: 0, wrong_value_2: 0; true value: 0.8, wrong_value_1: 0.1, wrong_value_2: 0.1
    if(flag_multi_class):
        img = img / 255
        new_mask = np.zeros(mask[:,:,:,0].shape + (len(COLOR_DICT),))
        new_mask += wrong_value_rate
        
        for i in range(len(COLOR_DICT)):
            color = COLOR_DICT[i]
            r_mask = mask[:,:,:,0]
            g_mask = mask[:,:,:,1]
            b_mask = mask[:,:,:,2]
            label_mask = (r_mask > color[0]-THRESHOLD) * (r_mask < color[0]+THRESHOLD) * (g_mask > color[1]-THRESHOLD) * (g_mask < color[1]+THRESHOLD) * (b_mask > color[2]-THRESHOLD) * (b_mask < color[2] + THRESHOLD)
            
            new_mask[label_mask,i] = true_value # usually 1...
        mask = new_mask
        
        mask[mask.sum(axis=(3)) != 1, 0] = true_value #all instances, which are in no class, go to unlabeled
        
        if mask.sum() != (mask.shape[0]*mask.shape[1]*mask.shape[2]):
            print('multi-label one-hot encode: %d; actual pixels: %d'%(mask.sum(), (mask.shape[0]*mask.shape[1]*mask.shape[2])))
            
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = true_value
        mask[mask <= 0.5] = wrong_value
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1, true_value = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class, true_value=true_value)
        yield (img,mask)



def testGenerator(test_path,num_image = 30,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

#TODO: remove num_class
def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def get_color_label_matrix(label_matrix):
    color_label_matrix = np.zeros(label_matrix.shape + (3,), dtype=np.int16)
#     color_label_matrix = np.zeros(tuple(label_matrix.get_shape().as_list()) + (3,), dtype=np.int16)

    for i in range(len(COLOR_DICT)):
        c = COLOR_DICT[i]
        color_label_matrix[label_matrix == i] = c
        
    return color_label_matrix

def visualize_result(pred):
    label_matrix = pred.argmax(axis=(0,1))
    return labelVisualize(len(COLOR_DICT), COLOR_DICT, label_matrix)

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)