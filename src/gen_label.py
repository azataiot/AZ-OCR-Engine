import os
import re
import time
import progressbar
import pickle
import pprint
from read_label import *
kz_char_txt="src/kz_characters.txt" #将这里的txt 切换为你自己的训练字符。
print("\n*** AZ OCR info:  读取字符数据 ***\n")
def read_text_char(src):
    with open(src) as file_object:
        lines = file_object.read().split()
    lenth=len(lines)
    time.sleep(0.01)
    return lines,lenth
chars,total = read_text_char(kz_char_txt) # read the txt line by line
IDs = range(total)
ID_list=list(IDs)
print("\n*** QAZKAZ info:  总共 "+ str(total) +  "个字符数据导入成功! ***\n")
values = chars
keys=ID_list
labels = dict(zip(keys, values))
kz_labels= open('src/kz_labels', 'wb')
print("\n*** QAZKAZ info:  正在生成label数据:  ***\n")
for i in progressbar.progressbar(range(total)):
    pickle.dump(labels,kz_labels)
    time.sleep(0.02)
order=input("\n*** QAZKAZ info:  是否要查看已经生成的label文件 ? y/n: ")
if(order=='y'):
    read_label(kz_labels)
else:
    pass
