import sys
import time 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json

class_names = {}

image_size = 224
def process_image(numPyImage):
    img = tf.cast(numPyImage, tf.float32)
    img = tf.image.resize(img, (image_size, image_size))
    img /= 255
    npArray = img.numpy()
    return npArray

def predict(image_path, model, top_k=5):
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    
    imgProb = model.predict(image)
    
    k_values, k_indices = tf.nn.top_k(imgProb, k= top_k)
    k_values =k_values.numpy()
    k_indices = k_indices.numpy()
    
   
    return k_values, k_indices,
def predict_list(path, model, top_n=5):
    class_list= []
    prob, img_class = predict(path, model, top_n)
    prob = prob[0]
    img_class = img_class[0]+1
    for i in img_class:
        class_list.append(class_names[str(i)])
    return prob, class_list 
    
        
if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    print(args)
    
    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    image_path = args.arg1
    mod = args.arg2
    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k
    if top_k is None: 
        top_k = 5
    if args.category_names is None:
        with open('label_map.json', 'r') as f:
            class_names = json.load(f)
    else:    
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
    #open('label_map.json', 'r')
    probs, classes = predict_list(image_path, model, top_k)
    
    print(probs)
    print(classes)

