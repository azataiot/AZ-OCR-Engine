import os
import glob
import time
import progressbar
import shutil

def get_available_fonts():
    """准备字体文件，并把字体文件复制到项目目录以便调用。
    :return list:
    """
    print("\n*** AZ OCR INFO: 正在加载可用的字体列表：***\n")
    os_fonts=glob.glob('/Library/Fonts/*')
    print("\n*** AZ OCR INFO: 加载完成***\n")
    return os_fonts

def display_available_fonts():
    available_fonts=get_available_fonts()
    total=len(available_fonts)
    print("\n*** AZ OCR INFO:总共有"+str(total)+"个字体可以使用***\n")
    for font in available_fonts:
        print(font)
    print("\n*** AZ OCR INFO:总共有"+str(total)+"个字体可以使用***\n")

def cp_fonts(src_txt):
    """
    这里读取保存有字体列表的txt文件，txt文件里按行单个字体名称。包括字体后缀，比如 arial.ttf。
    :param str src_txt: 字体名称txt文件的所在地址
    :return
    """
    f = open(src_txt)
    fonts = f.readlines() 
    available_fonts=get_available_fonts()  
    font_dir="src/fonts/"   #读取全部内容 ，并以列表方式返回
    for font in fonts:
        if(font in available_fonts):
            shutil.move(font,font_dir)
        else:
            print("你想要的字体不在系统中！请先安装所需字体后再试！")

cp_fonts("src/fonts.txt")
# txt=open("src/fonts.txt")