import os
import glob
import shutil

def move_images():
    pngs=glob.glob('src/segment_pic/*png')
    PNGs=glob.glob('src/segment_pic/*PNG')
    jpgs=glob.glob('src/segment_pic/*jpg')
    JPGs=glob.glob('src/segment_pic/*JPG')
    jpegs=glob.glob('src/segment_pic/*jpeg')
    JPEGs=glob.glob('src/segment_pic/*JPEG')
    images=pngs+PNGs+jpegs+jpgs+JPEGs+JPGs
    print(images)
move_images()