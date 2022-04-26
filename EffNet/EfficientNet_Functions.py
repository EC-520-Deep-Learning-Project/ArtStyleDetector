import os
import sys
import tensorflow.compat.v1 as tf

import tensorflow_hub as hub
import itertools
import os

import matplotlib.pylab as plt
import numpy as np
import csv
from PIL import Image, ImageFile
import tensorflow as tf
#import tensorflow_datasets as tfds
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import effnetv2_model
from keras.utils.layer_utils import count_params

Image.MAX_IMAGE_PIXELS = 1000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_wikidata(train_path, val_path, test_path, batch_size = 4, image_size = 223):
    print("Loading data...")
    print()
    datagen_kwargs = dict(rescale=1./255)
    dataflow_kwargs = dict(target_size=(image_size, image_size),batch_size=batch_size, interpolation="bilinear")
    
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
    valid_generator = valid_datagen.flow_from_directory(val_path, shuffle=False, **dataflow_kwargs)
    
    train_datagen = valid_datagen
    train_generator = train_datagen.flow_from_directory(train_path, shuffle=True, **dataflow_kwargs)
    
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_path, shuffle=False, **dataflow_kwargs)
    
    label_map = (train_generator.class_indices)
    print("Label map:")
    print(label_map)
    
    return train_generator, valid_generator, test_generator


def get_class_string_from_index(index, generator):
    for class_string, class_index in generator.class_indices.items():
        if class_index == index:
            return class_string
        

        
def visualize_input(generator):
    plt.figure(figsize=(10, 10))

    for i in range(4):
        image, label = next(generator)
        image = image[0, :, :, :]
        index = np.argmax(label[0])
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title(get_class_string_from_index(index, generator))
        plt.axis("off")
        

#func to get effnet model
def unfreeze_effnet(model, num_unfreeze):
    n = len(model.layers)
    num_unfreeze = 5
    start_freeze = n - num_unfreeze
    for i in range(start_freeze, n):
        model.layers[i].trainable = True
    print("Unfroze {} layers".format(num_unfreeze))
    
def get_effnetv2(do_fine_tuning, model_name, weights = 'imagenet', unfreeze = 0):
    tf.keras.backend.clear_session()
    base_model = effnetv2_model.get_model(model_name, weights = weights, include_top=False)
    base_model.trainable = do_fine_tuning
    if unfreeze > 0:
        unfreeze_effnet(base_model, unfreeze)
        
    return base_model

#func to get overall model
def get_new_model(unfreeze = 0):
    model_name = 'efficientnetv2-l' #@param {type:'string'}
    do_fine_tuning = False
    weights = 'imagenet21k'
    image_size = 223
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[image_size, image_size, 3]),
        get_effnetv2(do_fine_tuning, model_name, weights = weights, unfreeze = unfreeze),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(25, activation='softmax'),
    ])
    model.build((None, image_size, image_size, 3))
    model.summary()
    return model


#restore model from checkpoint
def restore_model(checkpoint_path, compile_m = True, learning_rate = 0.001, momentum = 0.9):
    model_new = get_new_model()
    
    if compile_m:
        model_new.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), 
          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
          metrics=['accuracy', 'top_k_categorical_accuracy'])
        
    model_new.load_weights(checkpoint_path)
    return model_new



#train model
def train_effnetv2(model, train_generator, valid_generator, num_epochs, learning_rate, decay = False,
                   restore = False,\
                   checkpoint_path = "/projectnb/dl523/students/nannkat/EC520/training/cp.ckpt",\
                   model_path = '/projectnb/dl523/students/nannkat/EC520/training/effnetv2_model', momentum = 0.9):
    
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Save checkpoints in case training session is cut off
    callback_list = []
    callback_list.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True, verbose=1))
    callback_list.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1))
    
    if not restore:
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum), 
          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
          metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size
    
    hist = model.fit(train_generator, epochs = num_epochs,\
                     validation_data = valid_generator,\
                     steps_per_epoch = steps_per_epoch,\
                     validation_steps = validation_steps, \
                     callbacks = callback_list).history
    
    
    
    #save model as a whole to share with the others
    model.save(model_path)
    
    return model, hist

#plot loss and accuracy
def plot_training(hist):
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    import matplotlib.pyplot as plt

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, label='Training acc')
    plt.plot(epochs, val_acc, label='Validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    
def visualize_results(model, generator):
    plt.figure(figsize=(10, 10))

    for i in range(4):
        image, label = next(generator)
        image = image[0, :, :, :]
        true_index = np.argmax(label[0])
        
        prediction_scores = model(np.expand_dims(image, axis=0))
        predicted_index = np.argmax(prediction_scores)
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title("True label: " + get_class_string_from_index(true_index, generator) +"\nPredicted label: " + get_class_string_from_index(predicted_index, generator))
        plt.axis("off")
        
import pandas as pd
def predict_and_save(model, test_generator, output_path):
    preds = []
    true_labels = []
    progbar = tf.keras.utils.Progbar(len(test_generator))

    print("Predicting....")
    for batch in range(len(test_generator)):
        images, labels = test_generator.next()
        for i in range(len(images)):
            image = images[i, :, :, :]
            label = np.argmax(labels[i])
            prediction_scores = model(np.expand_dims(image, axis=0))
            pred = np.argmax(prediction_scores)
            preds.append(pred)
            true_labels.append(label)
        progbar.update(batch)
    
    
    preds_df = pd.DataFrame(list(zip(preds, true_labels)),
               columns =['Prediction', 'True Label'])
    preds_df.to_csv(output_path, index=False, header = False)
    
    return preds, true_labels, preds_df