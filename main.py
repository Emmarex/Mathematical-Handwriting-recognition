import sys
from os import listdir
import numpy as np
import pickle
from PIL import Image
import cv2
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
        random_forest_classifier = RandomForestClassifier()
        random_forest_classifier.fit(X_train,Y_train)
        accuracy_score = random_forest_classifier.score(X_train,Y_train)
        # save classifier
        pickle.dump(random_forest_classifier,open("Model/random_forest_classifier.pkl",'wb'))
        print(f"Model Accuracy Score : {accuracy_score}")
        test_accuracy_score = random_forest_classifier.score(X_test,Y_test)
        print(f"Model Accuracy Score (Test) : {test_accuracy_score}")
    else :
        print("An error occurred.")

def transform_to_latex(handwritten_text):
    latex_symbols_dict = {
        '-' : '-',',' : ',','!' : '!','(' : '(',')' : ')',
        '[' : '[',']' : ']','{' : '{','}' : '}','+' : '+','=' : '=',
        '0' : '0','1' : '1','2' : '2','3' : '3','4' : '4','5' : '5',
        '6' : '6','7' : '7','8' : '8','9' : '9','A' : 'A','alpha' : '\\alpha',
        'ascii_124' : '|','b' : 'b','beta' : '\\beta','C' : 'C','cos' : '\\cos','d' : 'd',
        'Delta' : '\\delta','div' : '\\div','e' : 'e','exists' : '\\exists','f' : 'f','\\forall' : 'forall',
        'forward_slash' : '/','G' : 'G','gamma' : '\\gamma','geq' : '\\geq','gt' : '>','H' : 'H',
        'i' : 'i','in' : '\\in','infty' : '\\infty','int' : '\\int','j' : 'j','k' : 'k',
        'l' : 'l','lambda' : '\\lambda','ldots' : '\\ldots','leq' : '\leq','\\lim' : 'lim','log' : '\log',
        'lt':'<','M':'M','mu':'\mu','N':'N','neq':'\\neq','o':'o','p':'p','phi':'\\phi','pi':'\\pi',
        'pm':'\\pm','prime':'\\prime','q':'q','R':'R','rightarrow':'\Rightarrow','S':'S','sigma':'\\sigma',
        'sin':'\\sin','sqrt':'\\sqrt','sum':'\sum','T':'T','tan':'\\tan','theta':'\\theta','times':'\\times',
        'u':'u','v':'v','w':'w','X':'X','y':'y','z':'z'
    }
    return latex_symbols_dict.get(handwritten_text)

def main():
    try:
        args = sys.argv
        if len(args) >= 2 :
            image_directory = args[1]
            image_file = [get_image_matrix(image_directory)]
            # load saved model
            try:
                saved_decision_tree_classifier_model = pickle.load(open("Model/random_forest_classifier.pkl",'rb'))
                model_prediction = saved_decision_tree_classifier_model.predict(image_file)
                print(f"Recognized Digit : {model_prediction[0]} \nLatex Equivalent : {transform_to_latex(model_prediction[0])}")
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