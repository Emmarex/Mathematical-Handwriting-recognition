import sys
from os import listdir
import numpy as np
import pickle
from PIL import Image
import cv2
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

default_image_size = tuple((45,45))

def get_image_matrix(image_dir):
    try:
        image_grayscale = Image.open(image_dir).convert('L')
        #resize image
        image_grayscale = image_grayscale.resize(default_image_size, Image.ANTIALIAS)
        #
        image_np = np.array(image_grayscale)
        img_list = []
        for line in image_np:
            for value in line:
                img_list.append(value)
        return img_list
    except Exception as e:
        print(f"Error : {e}")
        return None

def get_train_test_images_from_directory(dataset_dir):
    X_train, X_test, Y_train, Y_test = [], [], [], []
    try:
        directory_list = listdir(dataset_dir)
        # remove '.DS_Store' from list
        if '.DS_Store' in directory_list:
            directory_list.remove('.DS_Store')
        # remove empty directory
        for directory in directory_list:
            if (len(f"{dataset_dir}/{directory}")) < 1 :
                directory_list.remove(directory)
        # check for empty dataset folder
        if len(directory_list) < 1 :
            print("Train Dataset folder is empty or dataset folder contains no image")
            return None, None, None, None
        
        for directory in directory_list[:2]:
            print(directory)
            image_dir = listdir(f"{dataset_dir}/{directory}")
            if '.DS_Store' in image_dir:
                image_dir.remove('.DS_Store')
            
            split_point = int(0.9*len(image_dir))
            train_images, test_images = image_dir[:split_point], image_dir[split_point:]

            for images in train_images:
                X_train.append(get_image_matrix(f"{dataset_dir}/{directory}/{images}"))
                Y_train.append(directory)
            for images in test_images:
                X_test.append(get_image_matrix(f"{dataset_dir}/{directory}/{images}"))
                Y_test.append(directory)

        return X_train, X_test, Y_train, Y_test

    except Exception as e:
        print(f"Error : {e}")
        return None, None, None, None

def train_model():
    train_dataset_dir = "./Dataset/"
    X_train, X_test, Y_train, Y_test = get_train_test_images_from_directory(train_dataset_dir)
    if X_train is not None and X_test is not None and Y_train is not None and Y_test is not None :
        # decision_tree_classifier = tree.DecisionTreeClassifier()
        decision_tree_classifier = RandomForestClassifier()
        decision_tree_classifier.fit(X_train,Y_train)
        accuracy_score = decision_tree_classifier.score(X_train,Y_train)
        # save classifier
        pickle.dump(decision_tree_classifier,open("Model/decision_tree_classifier02.pkl",'wb'))
        print(f"Model Accuracy Score : {accuracy_score}")
        test_accuracy_score = decision_tree_classifier.score(X_test,Y_test)
        print(f"Model Accuracy Score (Test) : {test_accuracy_score}")
    else :
        print("An error occurred.")

def main():
    try:
        args = sys.argv
        if len(args) >= 2 :
            image_directory = args[1]
            image_file = [get_image_matrix(image_directory)]
            # load saved model
            try:
                saved_decision_tree_classifier_model = pickle.load(open("Model/decision_tree_classifier.pkl",'rb'))
                model_prediction = saved_decision_tree_classifier_model.predict(image_file)
                print(f"Recognized Digit : {model_prediction[0]}")
            except FileNotFoundError as model_file_error:
                print(f"Error : {model_file_error}")
                print("... Training Model")
                train_model()
        else:
            print("Error : You have not specified an image path")
    except FileNotFoundError as file_error:
        print(f"Error : {file_error}")
    except Exception as e:
        print(f"Error : {e}")

main()