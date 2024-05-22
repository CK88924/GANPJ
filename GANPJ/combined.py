# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:06:42 2024

@author: CK
"""

import os
from PIL import Image

# 定義輸入和目標圖像的路徑
input_dir = r"dataset\blurred"
target_dir =r"dataset\target"
output_dir =r"dataset\combined" 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 獲取輸入和目標圖像的文件名列表
input_images = sorted(os.listdir(input_dir))
target_images = sorted(os.listdir(target_dir))

for input_image_name, target_image_name in zip(input_images, target_images):
    # 打開輸入和目標圖像
    input_image = Image.open(os.path.join(input_dir, input_image_name)).convert('RGB')
    target_image = Image.open(os.path.join(target_dir, target_image_name)).convert('RGB')

    # 檢查圖像尺寸是否匹配
    if input_image.size != target_image.size:
        print(f"Image sizes do not match for {input_image_name} and {target_image_name}. Skipping.")
        continue

    # 創建新的圖像，寬度為兩個圖像的寬度之和，高度保持不變
    combined_width = input_image.width + target_image.width
    combined_image = Image.new('RGB', (combined_width, input_image.height))

    # 將輸入和目標圖像粘貼到新的圖像上
    combined_image.paste(input_image, (0, 0))
    combined_image.paste(target_image, (input_image.width, 0))

    # 保存合並後的圖像為JPG格式
    combined_image.save(os.path.join(output_dir, input_image_name.replace('.png', '_combined.jpg')))

print("Image combination completed.")
