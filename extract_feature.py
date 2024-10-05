import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
import os, csv
from keras_vggface import utils
import keras
import unittest
from keras.engine import  Model
import tensorflow as tf

#刚开始分配少量资源，然后按需慢慢增加GPU资源
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)


def extract_feature(path, layer_name,vgg_model):
    #vgg_model = VGGFace(model='resnet50')  # pooling: None, avg or max
    out = vgg_model.get_layer(layer_name).input
    vgg_model_flatten = Model(vgg_model.input, out)
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x[:, :, :, ::-1]
    vgg_model_fc7_preds = vgg_model_flatten.predict(x=x)
    feature_vector = vgg_model_fc7_preds[0]
    return feature_vector


def main(npy_dir, image_dir):
    vgg_model = VGGFace(model='resnet50')
    image_paths = sorted(os.listdir(image_dir))

    count = 0
    for image_path in image_paths:
        image_path_all = os.path.join(image_dir, image_path)
        feature_vector = extract_feature(image_path_all, 'classifier', vgg_model)
        feature_vector = feature_vector.reshape(2048, 1)
        # 默认图片文件为.png后缀，若为其他后缀请在此处更改，否则无法保存npy文件
        npy_path = os.path.join(npy_dir, image_path)
        npy_path = npy_path.replace('.png', '.npy')
        np.save(npy_path, feature_vector)
        count = count + 1
        print(npy_path + " was extracted!  %d done!" % count)

