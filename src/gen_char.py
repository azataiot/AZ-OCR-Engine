# -*- coding: utf-8 -*-

from PIL import Image # image handling pack
from PIL import ImageFont
from PIL import ImageDraw
import pickle 
import pprint #pretty print ,print objects in line one line and one 
import fnmatch # file name match .returns bool . just check the name of the file is the same or not.
import os
import cv2
import json
import random
import numpy as np 
import shutil # advanced file processing pack
import copy #just used to copy the data 
import traceback
import argparse
from argparse import RawTextHelpFormatter



class dataAugmentation(object):  #Rotation/reflection/flip/zoom/shift/scale/contrast/noise/color
    def __init__(self,noise=True,dilate=True,erode=True):
        self.noise=noise
        self.dilate=dilate
        self.erode=erode

    @classmethod
    def add_noise(cls,img):
        for i in range(20): #add noise
            temp_x=np.random.randint(0,img.shape[0])
            temp_y=np.random.randint(0,img.shape[1])
            img[temp_x][temp_y] =255
        return img

    @classmethod
    def add_erode(cls,img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))    
        img = cv2.erode(img,kernel) 
        return img

    @classmethod
    def add_dilate(cls,img):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))    
        img = cv2.dilate(img,kernel) 
        return img

    def do(self,img_list=[]):
        aug_list= copy.deepcopy(img_list)
        for i in range(len(img_list)):
            im = img_list[i]
            if self.noise and random.random()<0.5:
                im = self.add_noise(im)
            if self.dilate and random.random()<0.5:
                im = self.add_dilate(im)
            elif self.erode:
                im = self.add_erode(im)    
            aug_list.append(im)
        return aug_list


# Scale the font image proportionally
class PreprocessResizeKeepRatio(object):

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def do(self, cv2_img):
        max_width = self.width
        max_height = self.height

        cur_height, cur_width = cv2_img.shape[:2]

        ratio_w = float(max_width)/float(cur_width)
        ratio_h = float(max_height)/float(cur_height)
        ratio = min(ratio_w, ratio_h)

        new_size = (min(int(cur_width*ratio), max_width),
                    min(int(cur_height*ratio), max_height))
        new_size = (max(new_size[0], 1),
                    max(new_size[1], 1),)

        resized_img = cv2.resize(cv2_img, new_size)
        return resized_img


# Find the smallest inclusion rectangle of a font
class FindImageBBox(object):
    def __init__(self, ):
        pass
    
    def do(self, img):
        height = img.shape[0]
        width = img.shape[1]
        v_sum = np.sum(img, axis=0)
        h_sum = np.sum(img, axis=1)
        left = 0
        right = width - 1
        top = 0
        low = height - 1
        # Scan from left to right, encountering a non-zero pixel as the left border of the font
        for i in range(width):
            if v_sum[i] > 0:
                left = i
                break
        
        for i in range(width - 1, -1, -1):
            if v_sum[i] > 0:
                right = i
                break

        for i in range(height):
            if h_sum[i] > 0:
                top = i
                break
        for i in range(height - 1, -1, -1):
            if h_sum[i] > 0:
                low = i
                break

        
        return (left, top, right, low)


class PreprocessResizeKeepRatioFillBG(object):

    def __init__(self, width, height,
                 fill_bg=False,
                 auto_avoid_fill_bg=True,
                 margin=None):
        self.width = width
        self.height = height
        self.fill_bg = fill_bg
        self.auto_avoid_fill_bg = auto_avoid_fill_bg
        self.margin = margin
        

    @classmethod
    def is_need_fill_bg(cls, cv2_img, th=0.5, max_val=255):
        image_shape = cv2_img.shape
        height, width = image_shape
        if height * 3 < width:
            return True
        if width * 3 < height:
            return True
        return False


    @classmethod
    def put_img_into_center(cls, img_large, img_small, ):
        width_large = img_large.shape[1]
        height_large = img_large.shape[0]

        width_small = img_small.shape[1]
        height_small = img_small.shape[0]

        if width_large < width_small:
            raise ValueError("width_large <= width_small")
        if height_large < height_small:
            raise ValueError("height_large <= height_small")

        start_width = (width_large - width_small) / 2
        start_height = (height_large - height_small) / 2

        img_large[start_height:start_height + height_small,
                  start_width:start_width + width_small] = img_small

        return img_large

    def do(self, cv2_img):
		# 
        if self.margin is not None:
            width_minus_margin = max(2, self.width - self.margin)
            height_minus_margin = max(2, self.height - self.margin)
        else:
            width_minus_margin = self.width
            height_minus_margin = self.height

        cur_height, cur_width = cv2_img.shape[:2]
        if len(cv2_img.shape) > 2:
            pix_dim = cv2_img.shape[2]
        else:
            pix_dim = None

        preprocess_resize_keep_ratio = PreprocessResizeKeepRatio(
            width_minus_margin,
            height_minus_margin)
        resized_cv2_img = preprocess_resize_keep_ratio.do(cv2_img)

        if self.auto_avoid_fill_bg:
            need_fill_bg = self.is_need_fill_bg(cv2_img)
            if not need_fill_bg:
                self.fill_bg = False
            else:
                self.fill_bg = True


        ## should skip horizontal stroke
        if not self.fill_bg:
            ret_img = cv2.resize(resized_cv2_img, (width_minus_margin,
                                                   height_minus_margin))
        else:
            if pix_dim is not None:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((height_minus_margin,
                                     width_minus_margin),
                                    np.uint8)
			# Place the scaled font image in the center of the background image
            ret_img = self.put_img_into_center(norm_img, resized_cv2_img)

        if self.margin is not None:
            if pix_dim is not None:
                norm_img = np.zeros((self.height,
                                     self.width,
                                     pix_dim),
                                    np.uint8)
            else:
                norm_img = np.zeros((self.height,
                                     self.width),
                                    np.uint8)
            ret_img = self.put_img_into_center(norm_img, ret_img)
        return ret_img   

# check the fonts availablity
class FontCheck(object):

    def __init__(self, lang_chars, width=32, height=32):
        self.lang_chars = lang_chars
        self.width = width
        self.height = height

    def do(self, font_path):
        width = self.width
        height = self.height
        try:
            for i, char in enumerate(self.lang_chars):
                img = Image.new("RGB", (width, height), "black") # black fonts
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype(font_path, int(width * 0.9),)
                # white fonts
                draw.text((0, 0), char, (255, 255, 255),
                          font=font)
                data = list(img.getdata())
                sum_val = 0
                for i_data in data:
                    sum_val += sum(i_data)
                if sum_val < 2:
                    return False
        except:
            print("fail to load:%s" % font_path)
            traceback.print_exc(file=sys.stdout)
            return False
        return True



# generate the pictures
class Font2Image(object):

    def __init__(self,
                 width, height,
                 need_crop, margin):
        self.width = width
        self.height = height
        self.need_crop = need_crop
        self.margin = margin

    def do(self, font_path, char, rotate=0):
        find_image_bbox = FindImageBBox()
        # black back
        img = Image.new("RGB", (self.width, self.height), "black")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, int(self.width * 0.7),)
        # white font
        draw.text((0, 0), char, (255, 255, 255),
                  font=font)
        if rotate != 0:
            img = img.rotate(rotate)
        data = list(img.getdata())
        sum_val = 0
        for i_data in data:
            sum_val += sum(i_data)
        if sum_val > 2:
            np_img = np.asarray(data, dtype='uint8')
            np_img = np_img[:, 0]
            np_img = np_img.reshape((self.height, self.width))
            cropped_box = find_image_bbox.do(np_img)
            left, upper, right, lower = cropped_box
            np_img = np_img[upper: lower + 1, left: right + 1]
            if not self.need_crop:
                preprocess_resize_keep_ratio_fill_bg = \
                    PreprocessResizeKeepRatioFillBG(self.width, self.height,
                                                    fill_bg=False,
                                                    margin=self.margin)
                np_img = preprocess_resize_keep_ratio_fill_bg.do(
                    np_img)
            # cv2.imwrite(path_img, np_img)
            return np_img
        else:
            print("img doesn't exist.")

# Attention: inside the kz_labels document, data ordered as : (ID:Character)
def get_label_dict():
    f=open('ocr/local/kz_labels','r')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

def args_parse():
    #input parser
    parser = argparse.ArgumentParser(
        description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--out_dir', dest='out_dir',
                        default=None, required=True,
                        help='write a caffe dir')
    parser.add_argument('--font_dir', dest='font_dir',
                        default=None, required=True,
                        help='font dir to to produce images')
    parser.add_argument('--test_ratio', dest='test_ratio',
                        default=0.2, required=False,
                        help='test dataset size')
    parser.add_argument('--width', dest='width',
                        default=None, required=True,
                        help='width')
    parser.add_argument('--height', dest='height',
                        default=None, required=True,
                        help='height')
    parser.add_argument('--no_crop', dest='no_crop',
                        default=True, required=False,
                        help='', action='store_true')
    parser.add_argument('--margin', dest='margin',
                        default=0, required=False,
                        help='', )
    parser.add_argument('--rotate', dest='rotate',
                        default=0, required=False,
                        help='max rotate degree 0-45')
    parser.add_argument('--rotate_step', dest='rotate_step',
                        default=0, required=False,
                        help='rotate step for the rotate angle')
    parser.add_argument('--need_aug', dest='need_aug',
                        default=False, required=False,
                        help='need data augmentation', action='store_true')   
    args = vars(parser.parse_args()) 
    return args
if __name__ == "__main__":

    description = '''
python gen_printed_char.py --out_dir ./dataset \
			--font_dir ./chinese_fonts \
			--width 30 --height 30 --margin 4 --rotate 30 --rotate_step 1
    '''
    options = args_parse()

    out_dir = os.path.expanduser(options['out_dir'])
    font_dir = os.path.expanduser(options['font_dir'])
    test_ratio = float(options['test_ratio'])
    width = int(options['width'])
    height = int(options['height'])
    need_crop = not options['no_crop']
    margin = int(options['margin'])
    rotate = int(options['rotate'])
    need_aug = options['need_aug']
    rotate_step = int(options['rotate_step'])
    train_image_dir_name = "train"
    test_image_dir_name = "test"

    # sepretly save the dataset to train_images_dir and the test_images_dir
    train_images_dir = os.path.join(out_dir, train_image_dir_name)
    test_images_dir = os.path.join(out_dir, test_image_dir_name)

    if os.path.isdir(train_images_dir):
        shutil.rmtree(train_images_dir)
    os.makedirs(train_images_dir)

    if os.path.isdir(test_images_dir):
        shutil.rmtree(test_images_dir)
    os.makedirs(test_images_dir)
    
    #get read the label file and get the character and the ID
    label_dict = get_label_dict()
    
    char_list=[]  # character list
    value_list=[] # label list
    for (value,chars) in label_dict.items():
        print (value,chars)
        char_list.append(chars)
        value_list.append(value)

    # new label order：（character：ID）
    lang_chars = dict(zip(char_list,value_list)) 
    font_check = FontCheck(lang_chars) 
    
    if rotate < 0:
        roate = - rotate
    
    if rotate > 0 and rotate <= 45:
        all_rotate_angles = []
        for i in range(0, rotate+1, rotate_step):  
            all_rotate_angles.append(i)
        for i in range(-rotate, 0, rotate_step):
            all_rotate_angles.append(i)
        #print(all_rotate_angles)
    
    # 对于每类字体进行小批量测试
    verified_font_paths = []
    ## search for file fonts
    for font_name in os.listdir(font_dir):
        path_font_file = os.path.join(font_dir, font_name)
        if font_check.do(path_font_file):
            verified_font_paths.append(path_font_file)

    font2image = Font2Image(width, height, need_crop, margin)

    for (char, value) in lang_chars.items():  # loop for character
        image_list = []
        print (char,value)
        #char_dir = os.path.join(images_dir, "%0.5d" % value)
        for j, verified_font_path in enumerate(verified_font_paths):    # loop for the font  
            if rotate == 0:
                image = font2image.do(verified_font_path, char)
                image_list.append(image)
            else:
                for k in all_rotate_angles:	
                    image = font2image.do(verified_font_path, char, rotate=k)
                    image_list.append(image)


        if need_aug:
            data_aug = dataAugmentation()
            image_list = data_aug.do(image_list)
            
        test_num = len(image_list) * test_ratio
        random.shuffle(image_list)  # pictures in unordered order
        count = 0
        for i in range(len(image_list)):
            img = image_list[i]
            #print(img.shape)
            if count < test_num :
                char_dir = os.path.join(test_images_dir, "%0.5d" % value)
            else:
                char_dir = os.path.join(train_images_dir, "%0.5d" % value)

            if not os.path.isdir(char_dir):
                os.makedirs(char_dir)

            path_image = os.path.join(char_dir,"%d.png" % count)
            cv2.imwrite(path_image,img)
            count += 1



# python gen_char.py --out_dir ocr/local/dataset/ --font_dir ocr/local/generate/kz_fonts_simple/ --width 30 --height 30 --margin 4 --rotate 30 --rotate_step 1