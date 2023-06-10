import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV



class AudioClassifier:
    """
    This class is used to classify audio files into cat and dog.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.categories = ['cat','dog']
        self.training_data = []
        self.X = []
        self.y = []
        self.lenofaudio = 0
        self.create_training_data()

    def create_training_data(self):
        for category in self.categories:
            path=os.path.join(self.data_dir, category)
            class_num=self.categories.index(category)
            for audio in os.listdir(path):
                audio_array, sr_array=librosa.load(os.path.join(path,audio))
                audio_array = audio_array[0:20765]
                self.training_data.append([audio_array,class_num])
        self.lenofaudio = len(self.training_data)
        print("Length of training dataset_sites: ", self.lenofaudio)

    def split_data(self):
        for categories, label in self.training_data:
            self.X.append(categories)
            self.y.append(label)
        self.X = np.array(self.X).reshape(self.lenofaudio,-1)
        self.y = np.array(self.y)
        print("Shape of X: ", self.X.shape)
        print("Shape of y: ", self.y.shape)

    def train_test_split_data(self):
        return train_test_split(self.X,self.y)

    def classify_data(self, X_train, X_test, y_train, y_test):
        classifiers = [DecisionTreeClassifier(), GaussianNB(), SVC(kernel='sigmoid',gamma='auto'), KNeighborsClassifier()]
        for clf in classifiers:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            print("Classifier: ", clf.__class__.__name__)
            print("Accuracy: ", accuracy_score(y_test, y_pred))
            print("Classification report: ", classification_report(y_test, y_pred))
            print("--------------------------------------------------")

    def cross_validate(self, X_train, y_train):
        knn_cv = KNeighborsClassifier()
        cv_scores = cross_val_score (knn_cv, X_train, y_train, cv = 50)
        print("Cross validation scores: ", cv_scores)
        print("Mean cross validation score: ", np.mean(cv_scores))

    def grid_search_validate(self, X_train, y_train):
        knn2 = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 5)}
        knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
        knn_gscv.fit(X_train, y_train)
        print("Best params: ", knn_gscv.best_params_)
        print("Best score: ", knn_gscv.best_score_)

    def run(self):
        self.split_data()
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        self.classify_data(X_train, X_test, y_train, y_test)
        self.cross_validate(X_train, y_train)
        self.grid_search_validate(X_train, y_train)

audioClassifier = AudioClassifier('cats_dogs/train')
audioClassifier.run()
