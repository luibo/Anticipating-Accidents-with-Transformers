#import cv2
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import logging, os
import utilities as utils
from encoder import Encoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow.compat.v1 as tf1
import tensorflow as tf
#tf1.disable_v2_behavior()

############### Global Parameters ###############
# path
train_path = '/home/l.borchia/dataset/features/training/'
test_path = '/home/l.borchia/dataset/features/testing/'
demo_path = '/home/l.borchia/dataset/features/testing/'
default_model_path = './tmp/video_classifier.weights.h5'
model_path = './tmp'
save_path = './model/'
video_path = '/home/l.borchia/dataset/videos/testing/positive/'

# batch_number
train_num = 128
test_num = 46

# Network Parameters
n_input = 4096 # fc6 or fc7(1*4096)
n_detection = 20 # number of object of each image (include image features)
n_hidden = 512 # hidden layer num of LSTM
n_img_hidden = 256 # embedding image features 
n_att_hidden = 256 # embedding object features
n_classes = 2 # has accident or not
n_frames = 100 # number of frame in each video

# Parameters
learning_rate = 0.0001
n_epochs = 30
batch_size = 10
display_step = 10

# Transformer parameters
num_layers = 4
d_model = 128
dff = 512
num_heads = 2
dropout_rate = 0.1

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 1024
IMG_SIZE = 128

EPOCHS = 20

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_transformer')
    parser.add_argument('--mode', dest = 'mode', help = 'train  (and test) or visualize', default = 'vis')
    parser.add_argument('--model', dest = 'model', default = default_model_path)
    parser.add_argument('--gpu', dest = 'gpu', default = '0')
    args = parser.parse_args()

    return args


def load_data(path, num_batch, mode):
    x = np.zeros((num_batch*batch_size, n_frames, n_detection, n_input), dtype = np.float16)
    y = np.zeros((num_batch*batch_size, n_classes), dtype = np.float16)
    z = np.zeros((num_batch*batch_size, n_frames, 19, 6), dtype = np.float16)
    k = np.full((num_batch*batch_size), '', dtype = "S10")
    for n in tqdm(np.arange(1, num_batch+1), desc=f'Loading {mode} batches'):
        file_name = '%03d' %n
        batch_data = np.load(path + 'batch_' + file_name + '.npz')
        data = batch_data['data']
        labels = batch_data['labels']
        det = batch_data['det']
        ID = batch_data['ID']
        x[(n-1)*batch_size:n*batch_size,:,:,:] = data
        y[(n-1)*batch_size:n*batch_size,:] = labels
        z[(n-1)*batch_size:n*batch_size,:,:,:] = det
        k[(n-1)*batch_size:n*batch_size] = ID
    y = np.argmax(y, axis=1)
    
    negatives = [x for x in y if x == 0]
    print(f'negatives: {len(negatives)}')

    positives = [x for x in y if x == 1]
    print(f'positives: {len(positives)}')

    return x, y, z, k


def get_compiled_model(shape):
    sequence_length = 100
    embed_dim = n_input
    dense_dim = 4

    inputs = tf.keras.Input(shape=shape)
    x = utils.PositionalEmbedding(
        sequence_length, n_detection, embed_dim, name="frame_position_embedding"
    )(inputs)
    x = Encoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    x = tf.keras.layers.GlobalMaxPooling2D(name="global_max_pooling")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
             "accuracy",
             tf.keras.metrics.Recall(),
             tf.keras.metrics.Precision()
        ],
    )

    return model


def run_experiment():
    # load data
    train_data, train_labels, _, _ = load_data(train_path, train_num, "training")
    test_data, test_labels, _, _ = load_data(test_path, test_num, "testing")

    # scale data
    scaler = StandardScaler()
    #test_data = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

    filepath = "./tmp/video_classifier.weights.h5"  
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=8, min_lr=0.000001
    )

    model = get_compiled_model(train_data.shape[1:])
    print(model.summary())

    #history = model.fit(
    #    train_data,
    #    train_labels,
    #    validation_split=0.15,
    #    epochs=EPOCHS,
    #    callbacks=[reduce],
    #)
    #print(history.history.keys())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        iteration = 1

        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]

            print(batch_data.shape)
            print(iteration)
            iteration += 1

            batch_data = scaler.fit_transform(batch_data.reshape(-1, batch_data.shape[-1])).reshape(batch_data.shape)

            model.train_on_batch(batch_data, batch_labels)

        """ loss, accuracy, recall, precision = model.evaluate(test_data, test_labels)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}") """

    
    model.save(model_path, save_format='tf')
    model = tf.keras.models.load_model(model_path)
    _, accuracy, recall, precision = model.evaluate(test_data, test_labels)
    print_metrics(accuracy, recall, precision)

    return model


def test_model():
    # load data
    test_data, test_labels, det, id = load_data(test_path, test_num, "testing")

    # restore model
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    _, accuracy, recall, precision = model.evaluate(test_data, test_labels)
    print_metrics(accuracy, recall, precision)


def print_predictions():
    # load data
    test_data, test_labels, det, id = load_data(test_path, test_num, "testing")

    # restore model
    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(test_data)
    print(predictions)


def print_metrics(accuracy, recall, precision):
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test recall: {round(recall * 100, 2)}%")
    print(f"Test precision: {round(precision * 100, 2)}%")
    

if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.mode == 'train':
           trained_model = run_experiment()
    elif args.mode == 'test':
           test_model()
    elif args.mode == 'vis':
           vis(args.model)


