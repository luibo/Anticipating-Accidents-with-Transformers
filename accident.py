#import cv2
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import logging, os
import utilities as utils
from encoder import Encoder
from tqdm import tqdm

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
    return x, y, z, k


def vis(checkpoint_path):
    # load data
    data, labels, det, id = load_data(test_path, 5, "visualization")

    # restore model
    model = tf.keras.models.load_model(model_path)
    print(model.summary())

    # run result
    #file_list = sorted(os.listdir(video_path))
    for i in labels:
        if i == 1 :
            pred = model.predict(data[i,:,:,:].reshape(1,100,20,4096))
            plt.figure(figsize=(14,5))
            plt.plot(pred, linewidth = 3.0)
            plt.ylim(0, 1)
            plt.ylabel('Probability')
            plt.xlabel('Frame')
            plt.title('Prediction')
            plt.show()
            file_name = id[i].decode('UTF-8')
            bboxes = det[i]
        #    new_weight = weight[:,:,i] * 255
        #    counter = 0
            #cap = cv2.VideoCapture(video_path + file_name + '.mp4')
            ret, frame = cap.read()

            while(ret):
                attention_frame = np.zeros((frame.shape[0],frame.shape[1]), dtype = np.uint8)
        #        now_weight = new_weight[counter,:]
        #        new_bboxes = bboxes[counter,:,:]
        #        index = np.argsort(now_weight)
        #        for num_box in index:
        #            if now_weight[num_box] / 255.0 > 0.4:
        #                cv2.rectangle(np.array(frame), (new_bboxes[num_box,0].astype(int), new_bboxes[num_box,1].astype(int)), (new_bboxes[num_box, 2].astype(int), new_bboxes[num_box, 3].astype(int)), (0, 255, 0), 3)
        #            else:
        #                cv2.rectangle(np.array(frame), (new_bboxes[num_box,0].astype(int), new_bboxes[num_box,1].astype(int)), (new_bboxes[num_box, 2].astype(int), new_bboxes[num_box, 3].astype(int)), (255, 0, 0), 2)
        #            font = cv2.FONT_HERSHEY_SIMPLEX
        #            cv2.putText(frame, str(round(now_weight[num_box] / 255.0 * 10000) / 10000), (new_bboxes[num_box, 0].astype(int), new_bboxes[num_box, 1].astype(int)), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #            attention_frame[int(new_bboxes[num_box, 1]):int(new_bboxes[num_box, 3]), int(new_bboxes[num_box, 0]):int(new_bboxes[num_box, 2])] = now_weight[num_box]
#
        #        attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
        #        dst = cv2.addWeighted(frame,0.6,attention_frame,0.4,0)
        #        cv2.putText(dst,str(counter+1),(10,30), font, 1,(255,255,255),3)
        #        cv2.imshow('result',dst)
        #        c = cv2.waitKey(50)
        #        ret, frame = cap.read()
        #        if c == ord('q') and c == 27 and ret:
        #            break
        #        counter += 1
        #    
        #cv2.destroyAllWindows()


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
    filepath = "./tmp/video_classifier.weights.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=8, min_lr=0.000001
    )

    model = get_compiled_model(train_data.shape[1:])
    print(model.summary())

    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[reduce],
    )

    print(history.history.keys())
    
    model.save(model_path, save_format='tf')
    model = tf.keras.models.load_model(model_path)
    _, accuracy = model.evaluate(test_data, test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return model


def test_model():
    # load data
    test_data, test_labels, det, id = load_data(test_path, 5, "visualization")

    # restore model
    model = tf.keras.models.load_model(model_path)
    print(model.summary())
    x, accuracy = model.evaluate(test_data, test_labels)




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


