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

def blur_license_plate(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    ROI = image[ymin:ymax, xmin:xmax]
    blurred_ROI = cv2.GaussianBlur(ROI, (5, 5), 5)
    return blurred_ROI

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
            blurred_plate = blur_license_plate(image, bbox)
            resized_plate = cv2.resize(blurred_plate, (256, 256))
            cv2.imwrite(os.path.join(output_folder, base_name), resized_plate)

if __name__=='__main__':
    # 定義文件夾路徑
    image_folder = 'dataset/images'
    xml_folder = 'dataset/annotations'
    output_folder = 'dataset/blurred'
    process_images(image_folder, xml_folder, output_folder)
