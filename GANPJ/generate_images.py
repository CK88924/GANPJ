import tensorflow as tf
import os
from matplotlib import pyplot as plt

# 定義加載數據集的方法
def load_image(image_file, blurred_folder):
    blurred_image_path = tf.strings.join([blurred_folder, image_file], separator=os.sep)
    blurred_image = tf.io.read_file(blurred_image_path)
    blurred_image = tf.image.decode_jpeg(blurred_image)
    blurred_image = tf.cast(blurred_image, tf.float32)
    blurred_image = tf.image.resize(blurred_image, [256, 256])
    blurred_image = (blurred_image / 127.5) - 1
    return blurred_image

# 生成圖片的方法
def generate_images(model, test_input, output_dir, output_name):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(10, 10))
    
    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    
    if not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, output_name)
    plt.savefig(output_path)
    plt.close()

# 主函數
if __name__ == '__main__':
    dataset_path = 'dataset'
    blurred_folder = os.path.join(dataset_path, 'blurred')
    checkpoint_dir = 'training_checkpoints'
    
    # 加載生成器
    generator = tf.keras.models.load_model(os.path.join(checkpoint_dir, 'generator.h5'))
    
    # 獲取測試圖像文件名
    test_files = sorted(os.listdir(blurred_folder))[:]  # 使用所有測試圖片
    
    output_dir = 'output_images'
    
    for image_file in test_files:
        input_image = load_image(image_file, blurred_folder)
        input_image = tf.expand_dims(input_image, 0)
        
        output_name = f"predicted_{os.path.splitext(image_file)[0]}.png"
        generate_images(generator, input_image, output_dir, output_name)
