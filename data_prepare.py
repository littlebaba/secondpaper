from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

IMAGE_SHAPE=(256,256)

def load_dataset(*path):
    """返回x_left、x_right、y"""
    args=len(path)
    
    x_left=[]
    x_right=[]
    y=[]
    
    if args ==0 or args >2:
        pass
    else:
        for x in sorted(os.listdir(path[0]),key=lambda x:x.split("_")[0][6:]):
            img=image.load_img(os.path.join(path[0],x))
            img=img.resize(IMAGE_SHAPE)
            img=image.img_to_array(img)
            img=img/255
            if 'left' in x:
                x_left.append(img)
            elif 'right' in x:
                x_right.append(img)
                
    if args ==1:
        return np.array(x_left),np.array(x_right)
    elif args ==2:
        for y_ in sorted(os.listdir(path[1]),key=lambda x:x.split("_")[0][6:]):
            img=image.load_img(os.path.join(path[1],y_))
            img=img.resize(IMAGE_SHAPE)
            img=image.img_to_array(img)
            img=img/255
            y.append(img)
        return np.array(x_left),np.array(x_right),np.array(y)
    

    
#     for x in sorted(os.listdir(x_path),key=lambda x:x.split("_")[0][6:]):
#         img=image.load_img(os.path.join(x_path,x))
#         img=img.resize(IMAGE_SHAPE)
#         img=image.img_to_array(img)
#         if 'left' in x:
#             x_left.append(img)
#         elif 'right' in x:
#             x_right.append(img)
    
#     for y_ in sorted(os.listdir(y_path),key=lambda x:x.split("_")[0][6:]):
#         img=image.load_img(os.path.join(y_path,y_))
#         img=img.resize(IMAGE_SHAPE)
#         img=image.img_to_array(img)
#         y.append(img)
#     return np.array(x_left),np.array(x_right),np.array(y)

def train_generate(batch_size,x_lf,x_rg,y_,aug_dict,lf_img_save_prefix="L",
                   rg_img_save_prefix="R",y_img_prefix="O",save_to_dir=None,seed=1):
    
    x_left_gen=ImageDataGenerator(**aug_dict)
    x_right_gen=ImageDataGenerator(**aug_dict)
    y_gen=ImageDataGenerator(**aug_dict)
    
    x_left_generator=x_left_gen.flow(
        x_lf,
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=lf_img_save_prefix)
    
    x_right_generator=x_right_gen.flow(
        x_rg,
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=rg_img_save_prefix)
    
    y_generator=y_gen.flow(
        y_,
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=y_img_prefix)
    
    train_generator=zip(x_left_generator,x_right_generator,y_generator)
    for (x_left,x_right,y) in train_generator:
        x_left=x_left
        x_right=x_right
        y=y
        yield([x_left,x_right],y)
        
def valid_generate(batch_size,x_lf,x_rg,y_,aug_dict,lf_img_save_prefix="L",
                   rg_img_save_prefix="R",y_img_prefix="O",save_to_dir=None,seed=1):
    
    x_left_gen=ImageDataGenerator(**aug_dict)
    x_right_gen=ImageDataGenerator(**aug_dict)
    y_gen=ImageDataGenerator(**aug_dict)
    
    x_left_generator=x_left_gen.flow(
        x_lf,
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=lf_img_save_prefix)
    
    x_right_generator=x_right_gen.flow(
        x_rg,
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=rg_img_save_prefix)
    
    y_generator=y_gen.flow(
        y_,
        batch_size=batch_size,
        seed=seed,
        save_to_dir=save_to_dir,
        save_prefix=y_img_prefix)
    
    train_generator=zip(x_left_generator,x_right_generator,y_generator)
    for (x_left,x_right,y) in train_generator:
        x_left=x_left
        x_right=x_right
        y=y
        yield([x_left,x_right],y)