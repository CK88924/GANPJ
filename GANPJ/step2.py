import cv2
import os
import xml.etree.ElementTree as ET
from glob import glob

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return (xmin, ymin, xmax, ymax)
    return None

def crop_and_resize(image, bbox, size=(256, 256), interpolation=cv2.INTER_LANCZOS4):
    xmin, ymin, xmax, ymax = bbox
    cropped_image = image[ymin:ymax, xmin:xmax]
    resized_image = cv2.resize(cropped_image, size, interpolation=interpolation)
    return resized_image

def process_images(image_folder, xml_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for image_path in glob(os.path.join(image_folder, '*.png')):  # 修改檔案類型如有需要
        base_name = os.path.basename(image_path)
        xml_path = os.path.join(xml_folder, base_name.replace('.png', '.xml'))

        image = cv2.imread(image_path)
        if image is None:
            continue

        bbox = parse_xml(xml_path)
        if bbox:
            resized_image = crop_and_resize(image, bbox)
            cv2.imwrite(os.path.join(output_folder, base_name), resized_image)

if __name__ == '__main__':
    # 定義文件夾路徑
    image_folder = 'dataset/images'
    xml_folder = 'dataset/annotations'
    output_folder = 'dataset/target'
    process_images(image_folder, xml_folder, output_folder)
