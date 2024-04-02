# -*- coding: utf-8 -*-
import os
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from PIL import ImageFont, ImageDraw, Image
import cv2
import re
import math
from unicodedata import normalize


# normalized file names
def nfd2nfc(data):
    return normalize("NFC", data)


image_path = 'data/data/testset_image'

test_path = 'data/data/testset_json'
test_json = [str(f) for f in Path(test_path).rglob('*.json')]
print(f'Size of testset: {len(test_json)}')


for each_file in tqdm(test_json):

    with open(each_file, encoding="utf-8") as f:
        data = json.load(f)

    file_name = data['meta']['file_name'].replace('-tf', '').replace('.jpg', '-tf.jpg')

    image_file = f"{image_path}/{file_name}"

    try:
        image = cv2.imread(image_file)
        image_width = image.shape[1]
        image_height = image.shape[0]
    
    except KeyboardInterrupt:
        sys.exit(0)

    except:
        print('ERROR can\'t load image :: ', image_file)
        continue

    for i, each_ocr in enumerate(data['annotations']):

        x = math.ceil(each_ocr['ocr']['x'] * image_width / 100)
        y = math.ceil(each_ocr['ocr']['y'] * image_height / 100)
        w = math.ceil(each_ocr['ocr']['width'] * image_width / 100)
        h = math.ceil(each_ocr['ocr']['height'] * image_height / 100)

        text = each_ocr['ocr']['text']

        if text.replace(' ', '') in ['개인정보', '메뉴판', '매뉴판', '메뉴명', '매뉴명', '비식별화']:
            # print(f'Skipped 비식별화 포함... :: {each_file.split("/")[-1]}\n')
            continue

        if 'points' in each_ocr['ocr']:
            # print(f'Skipped polygon 포함... :: {each_file.split("/")[-1]}\n')
            continue

        if each_ocr['ocr']['rotation'] != 0:
            label_x, label_y, label_w, label_h, label_r = (
                each_ocr['ocr']["x"],
                each_ocr['ocr']["y"],
                each_ocr['ocr']["width"],
                each_ocr['ocr']["height"],
                each_ocr['ocr']["rotation"]
            )

            r = math.radians(label_r)
            h_sin_r, h_cos_r = label_h * math.sin(r), label_h * math.cos(r)
            x_top_right = label_x + label_w * math.cos(r)
            y_top_right = label_y + label_w * math.sin(r)

            x_ls = [
                label_x,
                x_top_right,
                x_top_right - h_sin_r,
                label_x - h_sin_r
            ]
            y_ls = [
                label_y,
                y_top_right,
                y_top_right + h_cos_r,
                label_y + h_cos_r
            ]

            p1_new = (round(label_x * image.shape[1] / 100), round(label_y * image.shape[0] / 100))
            p2_new = (round(x_top_right * image.shape[1] / 100), round(y_top_right * image.shape[0] / 100))
            p3_new = (round((x_top_right - h_sin_r) * image.shape[1] / 100),
                      round((y_top_right + h_cos_r) * image.shape[0] / 100))
            p4_new = (round((label_x - h_sin_r) * image.shape[1] / 100), round((label_y + h_cos_r) * image.shape[0] / 100))

            roi_corners = np.array([[p1_new, p2_new, p3_new, p4_new]], dtype=np.int32)

            cropped_image = roi_corners

        cropped_image = image[y:y+h, x:x+w]


        try:
            resized_img = cv2.resize(cropped_image, dsize=(304, 64), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'{dataset_dir}/{nfd2nfc(text)}%{each_ocr["sn"]}.jpg', resized_img)

        except KeyboardInterrupt:
            sys.exit(0)

        except:
            print('Skipped.... ERROR can\'t save cropped_image :: ', image_file, '\n')
            continue


print('done!')
