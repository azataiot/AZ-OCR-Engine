# 先把我们需要的模块导入进来
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import progressbar
import glob
import shutil

# 准备需要切割的图片，这里我们假设可能会有很多图片的可能性，因此我们统一把所有的需要切割的图片放在一个 segmeng_pic 的目录.

# 读取文件夹内的图片文件。
# 读取文件夹内的图片文件
pngs=glob.glob('src/segment_pic/segment_int/*png')
PNGs=glob.glob('src/segment_pic/segment_int/*PNG')
jpgs=glob.glob('src/segment_pic/segment_int/*jpg')
JPGs=glob.glob('src/segment_pic/segment_int/*JPG')
jpegs=glob.glob('src/segment_pic/segment_int/*jpeg')
JPEGs=glob.glob('src/segment_pic/segment_int/*JPEG')

# 备注：为什么没有通过 glob（src/*。*） 这种方法导入所有图片呢？
# 这里又一个问题，因为我在Mac当中执行代码，有时候发现cv2 报错说 typeerror 原因是该文件夹内可以有非图片类型的文件，
# 或者可以有像苹果系统自动生成的隐藏的 .DS 文件，这个文件很恶心。所以我决定通过上面这种比较简陋，简单粗暴但是非常有效的方式读取文件夹内的
# 所有图片。
# 读取需要处理的照片。
#活的所有图片文件的绝对地址
images=pngs+PNGs+jpegs+jpgs+JPEGs+JPGs


base_dir = "src/segment_pic/segment_int/"
dst_dir = "src/segment_pic/segment_out/"
count=0
def extract_peek_ranges_from_array(array_vals, minimun_val=10, minimun_range=2):
    start_i = None
    end_i = None
    peek_ranges = []
    for i, val in enumerate(array_vals):
        if val > minimun_val and start_i is None:
            start_i = i
        elif val > minimun_val and start_i is not None:
            pass
        elif val < minimun_val and start_i is not None:
            end_i = i
            if end_i - start_i >= minimun_range:
                peek_ranges.append((start_i, end_i))
            start_i = None
            end_i = None
        elif val < minimun_val and start_i is None:
            pass
        else:
            raise ValueError("cannot parse this case...")
    return peek_ranges

def cutImage(img, peek_range):
    global count
    for i, peek_range in enumerate(peek_ranges):
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            w = vertical_range[1] - x
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            count += 1
            img1 = img[y:peek_range[1], x:vertical_range[1]]
            # new_shape = (150, 150)
            # img1 = cv2.resize(img1, new_shape)
            cv2.imwrite(dst_dir + str(count) + ".png", img1)
            # cv2.rectangle(img, pt1, pt2, color)

def median_split_ranges(peek_ranges):
    new_peek_ranges = []
    widthes = []
    for peek_range in peek_ranges:
        w = peek_range[1] - peek_range[0] + 1
        widthes.append(w)
    widthes = np.asarray(widthes)
    median_w = np.median(widthes)
    for i, peek_range in enumerate(peek_ranges):
        num_char = int(round(widthes[i]/median_w, 0))
        if num_char > 1:
            char_w = float(widthes[i] / num_char)
            for i in range(num_char):
                start_point = peek_range[0] + int(i * char_w)
                end_point = peek_range[0] + int((i + 1) * char_w)
                new_peek_ranges.append((start_point, end_point))
        else:
            new_peek_ranges.append(peek_range)
    return new_peek_ranges


for image in images:
    image_color = cv2.imread(image)
    new_shape = (image_color.shape[1] * 2, image_color.shape[0] * 2)
    image_color = cv2.resize(image_color, new_shape)
    image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
    print("\n*** AZ OCR INFO:正在生成 Binary 图片，请稍等...***\n")
    p=progressbar.ProgressBar()
    for i in p(range(100)):
        adaptive_threshold = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY_INV, 11, 2)
        time.sleep(0.02)
    cv2.imshow('binary image', adaptive_threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    horizontal_sum = np.sum(adaptive_threshold, axis=1)
    plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
    plt.gca().invert_yaxis()
    plt.show()

    print("\n*** AZ OCR INFO:正在画图，请稍等...***\n")
    peek_ranges = extract_peek_ranges_from_array(horizontal_sum)
    line_seg_adaptive_threshold = np.copy(adaptive_threshold)
    p=progressbar.ProgressBar()
    for i in p(range(100)):
        for i, peek_range in enumerate(peek_ranges):
            x = 0
            y = peek_range[0]
            w = line_seg_adaptive_threshold.shape[1]
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(line_seg_adaptive_threshold, pt1, pt2, 255)
            time.sleep(0.02)
    cv2.imshow('line image', line_seg_adaptive_threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek_ranges_from_array(
            vertical_sum,
            minimun_val=40,
            minimun_range=1)
        vertical_peek_ranges2d.append(vertical_peek_ranges)

    print("\n*** AZ OCR INFO:正在定位字符，请稍等...***\n")
    p=progressbar.ProgressBar()
    for i in p(range(100)):
        color = (0, 0, 255)
        for i, peek_range in enumerate(peek_ranges):
            for vertical_range in vertical_peek_ranges2d[i]:
                x = vertical_range[0]
                y = peek_range[0]
                w = vertical_range[1] - x
                h = peek_range[1] - y
                pt1 = (x, y)
                pt2 = (x + w, y + h)
                cv2.rectangle(image_color, pt1, pt2, color)
        time.sleep(0.01)
    cv2.imshow('char image', image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows


    p=progressbar.ProgressBar()
    vertical_peek_ranges2d = []
    for peek_range in peek_ranges:
        start_y = peek_range[0]
        end_y = peek_range[1]
        line_img = adaptive_threshold[start_y:end_y, :]
        vertical_sum = np.sum(line_img, axis=0)
        vertical_peek_ranges = extract_peek_ranges_from_array(
            vertical_sum,
            minimun_val=40,
            minimun_range=1)
        vertical_peek_ranges = median_split_ranges(vertical_peek_ranges)
        vertical_peek_ranges2d.append(vertical_peek_ranges)
    print("\n*** AZ OCR INFO:正在定位字符*，请稍等...***\n")
    olor = (0, 0, 255)
    for i, peek_range in enumerate(peek_ranges):
        for vertical_range in vertical_peek_ranges2d[i]:
            x = vertical_range[0]
            y = peek_range[0]
            w = vertical_range[1] - x
            h = peek_range[1] - y
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv2.rectangle(image_color, pt1, pt2, color)
    cv2.imshow('splited char image', image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    p=progressbar.ProgressBar()

    print("\n*** AZ OCR INFO:正在切割并保存图片，请稍等...***\n")
    cutImage(image, peek_range)
    for i in p(range(100)):
        time.sleep(0.01)
    print("\n*** AZ OCR INFO:Done！...***\n")


