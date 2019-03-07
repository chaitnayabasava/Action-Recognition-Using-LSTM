import numpy as np
import sys
import os


def main():
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.recurrent_networks import VGG16BidirectionalLSTMVideoClassifier
    from keras_video_classifier.library.utility.ucf.UCF101_loader import load_ucf, scan_ucf_with_labels

    vgg16_include_top = False
    data_dir_path = os.path.join(os.path.dirname(__file__), 'very_large_data')
    model_dir_path = os.path.join(os.path.dirname(__file__), 'models/giri')

    config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)
    weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(model_dir_path,
                                                                                  vgg16_include_top=vgg16_include_top)

    np.random.seed(42)

    load_ucf(data_dir_path)

    predictor = VGG16BidirectionalLSTMVideoClassifier()
    predictor.load_model(config_file_path, weight_file_path)

    videos = scan_ucf_with_labels(data_dir_path, [label for (label, label_index) in predictor.labels.items()])

    video_file_path_list = np.array([file_path for file_path in videos.keys()])
    np.random.shuffle(video_file_path_list)

    correct_count = 0
    count = 0

    for video_file_path in video_file_path_list:
        label = videos[video_file_path]
        correct_count, count = predictor.predict(video_file_path, label, correct_count, count)
        #print('predicted: ' + predicted_label + ' actual: ' + label)
        #print('Probability : %f'%(predicted_prob))
        #correct_count = correct_count + 1 if label == predicted_label else correct_count
        #count += 1
        #accuracy = correct_count / count
        #print('accuracy: ', accuracy)
        #break


if __name__ == '__main__':
    main()
