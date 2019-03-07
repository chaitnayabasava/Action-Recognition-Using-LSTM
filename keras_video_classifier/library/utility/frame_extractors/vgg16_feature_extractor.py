import random
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Lambda
from keras import backend as K
import cv2, numpy as np
import glob, os, re
from keras.activations import relu 
from keras.layers import Input, concatenate, Conv2DTranspose
from scipy.misc import imsave
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ModelCheckpoint
from  sklearn.model_selection import train_test_split
import imageio
from skimage import transform as tf
from scipy import ndimage
import scipy, skimage
from skimage.measure import compare_ssim, compare_psnr
from keras import losses
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import keras as keras


MAX_NB_CLASSES = 101

model1 = VGG16(include_top=False)
def loss_function(y_true, y_pred):
    #l = losses.mean_squared_error(y_true, y_pred)
    
    f_t = model1(y_true)
    f_p = model1(y_pred)	
    
    #l_fe = K.mean(K.square(f_p - f_t))
    l_fe = losses.mean_squared_error(f_t, f_p)
    return l_fe #+ l

def get_unet(arg1, trainable = True):
    inputs = Input(shape=(None,None, 3))
    conv1 = Conv2D(3, (1, 1), kernel_initializer='random_normal', activation='relu', trainable = trainable)(inputs)

    conv2 = Conv2D(3, (3, 3), kernel_initializer='random_normal', activation='relu',  padding='same', trainable = trainable)(conv1)

    concat1 = concatenate([conv1, conv2], axis=-1)

    conv3 = Conv2D(3, (5, 5), activation='relu', kernel_initializer='truncated_normal', padding='same', trainable = trainable)(concat1)

    concat2 = concatenate([conv2, conv3], axis=-1)

    conv4 = Conv2D(3, (7, 7), activation='relu', kernel_initializer='random_normal', padding='same', trainable = trainable)(concat2)

    concat3 = concatenate([conv1, conv2, conv3, conv4], axis=-1)

    K = Conv2D(3, (3, 3), activation='relu', kernel_initializer='truncated_normal', padding='same', trainable = True)(concat3)

    print (inputs.shape, K.shape)
    product= keras.layers.Multiply()([K, inputs])
    sum1 = keras.layers.Subtract()([product, K])
    sum2 = Lambda(lambda x: 1+x) (sum1)
    #sum2 = keras.layers.Add()([sum1, ones_tensor])
    out_layer = Lambda(lambda x: relu(x)) (sum2)
    ##out_layer = relu(sum2)#
    

    if arg1 == 1:
        model = Model(inputs=inputs,outputs=out_layer)
    else:
        model = Model(inputs=inputs,outputs=conv1)

    return model

def extract_vgg16_features_live(model, video_input_file_path, video_model, expected_frames, labels_idx2word, label, correct_count, count1):
    print('Extracting frames from video: ', video_input_file_path)

    from keras import optimizers
    rms = optimizers.RMSprop(lr=0.001, decay=0.0001, rho=0.9, )

    model1 = get_unet(1)
    model1.compile(loss=loss_function, optimizer=rms, metrics=['accuracy'])
    model1.load_weights('C:/Users/tanny/Desktop/keras-video-classifier-master/keras_video_classifier/library/utility/frame_extractors/aod_feature_model_try_new.h5')
    
    vidcap = cv2.VideoCapture(video_input_file_path)
    fps    = vidcap.get(cv2.CAP_PROP_FPS)
    print(fps)
    success, image = vidcap.read()
    features = []
    success = True
    count = 0
    text = ""
    while success:
        #vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            feature = model.predict(input).ravel()
            features.append(feature)
            count = count + 1

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, text, (5, 20), font, 0.5, (random.randint(0,255),random.randint(0,255),random.randint(0,255)), 2)
            cv2.imshow('Frame', img)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        if(count%20 == 0):
            features = np.array(features)
            frames = features.shape[0]
            if frames == 0:
                continue
            if frames > expected_frames:
                features = features[0:expected_frames, :]
            elif frames < expected_frames:
                print(features.shape)
                temp = np.zeros(shape=(expected_frames, features.shape[1]))
                temp[0:frames, :] = features
                features = temp

            predicted_class = np.argmax(video_model.predict(np.array([features]))[0])
            print(predicted_class)
            predicted_label = labels_idx2word[predicted_class]

            text = predicted_label + " - " + label
            """correct_count = correct_count + 1 if label == predicted_label else correct_count
            count1 += 1
            accuracy = correct_count / count1
            print('accuracy: ', accuracy)"""
            print(text)
            features = [] #list(features)
    #print(count)
    #unscaled_features = np.array(features)
    return correct_count, count1

def extract_vgg16_features(model, model1, video_input_file_path, feature_output_file_path):
    if os.path.exists(feature_output_file_path):
        return np.load(feature_output_file_path)
    count = 0
    print('Extracting frames from video: ', video_input_file_path)
    vidcap = cv2.VideoCapture(video_input_file_path)
    success, image = vidcap.read()
    features = []
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))  # added this line
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        if success:
            img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            input = img_to_array(img)
            input = np.expand_dims(input, axis=0)
            input = preprocess_input(input)
            out = model1.predict(input)
            feature = model.predict(out).ravel()
            features.append(feature)
            count = count + 1
    unscaled_features = np.array(features)
    np.save(feature_output_file_path, unscaled_features)
    return unscaled_features


def scan_and_extract_vgg16_features(data_dir_path, output_dir_path, model=None, data_set_name=None):
    if data_set_name is None:
        data_set_name = 'UCF-101'

    input_data_dir_path = data_dir_path + '/' + data_set_name
    output_feature_data_dir_path = data_dir_path + '/' + output_dir_path
    
    from keras import optimizers
    rms = optimizers.RMSprop(lr=0.001, decay=0.0001, rho=0.9, )

    model1 = get_unet(1)
    model1.compile(loss=loss_function, optimizer=rms, metrics=['accuracy'])
    model1.load_weights('C:/Users/tanny/Desktop/keras-video-classifier-master/keras_video_classifier/library/utility/frame_extractors/aod_feature_model_try_new.h5')
    if model is None:
        model = VGG16(include_top=True, weights='imagenet')
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    if not os.path.exists(output_feature_data_dir_path):
        os.makedirs(output_feature_data_dir_path)

    y_samples = []
    x_samples = []

    dir_count = 0
    for f in os.listdir(input_data_dir_path):
        file_path = input_data_dir_path + os.path.sep + f
        if not os.path.isfile(file_path):
            output_dir_name = f
            output_dir_path = output_feature_data_dir_path + os.path.sep + output_dir_name
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            dir_count += 1
            for ff in os.listdir(file_path):
                video_file_path = file_path + os.path.sep + ff
                output_feature_file_path = output_dir_path + os.path.sep + ff.split('.')[0] + '.npy'
                x = extract_vgg16_features(model, model1, video_file_path, output_feature_file_path)
                y = f
                y_samples.append(y)
                x_samples.append(x)

        if dir_count == MAX_NB_CLASSES:
            break

    return x_samples, y_samples

