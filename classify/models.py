import os
import datetime
import h5py
import logging
import argparse
import pickle
import pandas as pd
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from utils.file import generate_out_file
import config


class DNN_W:
    def __init__(
        self,
        train: tuple,
        valid: tuple,
        test: tuple,
        training_params: dict,
        out_dir: str,
        tag: str,
        encoder_file: str,
        lr: float = 0.00001,
        early_stopping: bool = True,
    ):

        self.save_dir = out_dir
        self.tag = tag
        self.encoder = self.init_encoder(encoder_file)

        self.X_train = train[0]
        self.X_valid = valid[0]
        self.X_test = test[0]

        self.y_train = train[1]
        self.y_valid = valid[1]
        self.y_test = test[1]

        self.training_params = training_params

        self.n_classes = self.y_train.shape[1]
        self.n_features = self.X_train.shape[1]
        self.model = self.init_model(self.n_features, self.n_classes, lr=lr)

        self.early_stopping = early_stopping

    def init_model(self, n_features, n_classes, lr):

        model = Sequential()
        model.add(Dense(2000, activation="relu", input_shape=(n_features,)))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(500, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(n_classes, activation="softmax"))

        opt = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(
            loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )
        return model

    def init_encoder(self, filepath: str):
        encoder = LabelEncoder()
        encoder.classes_ = np.load(filepath, allow_pickle=True)
        return encoder

    def fit(self, save: bool = True):

        # Initiate early stopping
        callbacks = []
        if self.early_stopping:
            early_stopping_monitor = EarlyStopping(patience=5)
            callbacks.append(early_stopping_monitor)

        # TODO: Fix tensorboard
        # Add TensorBoard
        # log_dir = config.TRAIN_LOGS + "/" + self.tag
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(
        #     log_dir=log_dir, histogram_freq=1
        # )
        # callbacks.append(tensorboard_callback)

        self.model.fit(
            self.X_train,
            self.y_train,
            callbacks=callbacks,
            validation_data=(self.X_valid, self.y_valid),
            **self.training_params,
        )

        if save:
            outfile = generate_out_file("sp_model.h5", self.save_dir, self.tag)
            self.model.save(outfile)
            print(f"Model saved to {outfile}")

    def predict(self):
        self.predictions = self.model.predict(self.X_test)

        self.predict_transform = self.encoder.inverse_transform(
            self.predictions.argmax(axis=1)
        )
        self.y_test_transform = self.encoder.inverse_transform(
            self.y_test.argmax(axis=1)
        )

        self.cm = metrics.confusion_matrix(
            self.y_test_transform, self.predict_transform
        )
        self.classification_report = metrics.classification_report(
            self.y_test_transform, self.predict_transform
        )
        self.auc = metrics.roc_auc_score(self.y_test, self.predictions)
        self.heatmap = sns.heatmap(
            self.cm / np.sum(self.cm), annot=True, fmt=".2%", cmap="Blues"
        )

    def report(self):
        print(self.classification_report)
        print(f"AUC: {self.auc}")
        heatmap_file = generate_out_file("confusion.png", self.save_dir, self.tag)
        self.heatmap.figure.savefig(heatmap_file)
        print(f"Confusion matrix saved to {heatmap_file}")


model_names = {
    "dnn_wide": DNN_W,
    # "knn": knn(),
    # "random_forest": random_forest(),
    # "svm": svm(),
    # "logistic_regression": logistic_regression(),
    # "decision_tree": decision_tree(),
}
# TODO Is dictionary necessary?
# TODO Add Cross Validation Option

