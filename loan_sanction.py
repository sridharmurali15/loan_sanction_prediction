import pickle
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


class Build():

    def __init__(self, inputfile):
        self.data = pd.read_csv(inputfile)
        self.label = self.data.iloc[:,-1]
        self.data.drop(['RowNumber','CustomerId','Surname','Exited', 'Geography'], axis=1, inplace=True)

        
    def DataProcess(self):
        self.data['Gender'] = self.data['Gender'].astype('category').cat.codes
        self.data.drop('IsActiveMember', axis=1, inplace=True)
        print(self.data)
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        
        xtr, xte, ytr, yte = train_test_split(self.data, self.label, stratify=self.label)

        sampler = RandomOverSampler()
        xtr, ytr = sampler.fit_resample(xtr, ytr)
        # xte, yte = sampler.fit_resample(xte, yte)
        return xtr, ytr #, xte, yte


    def BuildModel(self, X, y):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(32, input_shape=(X.shape[1],), activation='sigmoid'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(64, input_shape=(32,), activation='sigmoid'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(128, input_shape=(64,), activation='sigmoid'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(64, input_shape=(128,), activation='sigmoid'))
        model.add(tf.keras.layers.Dense(1, input_shape=(64,), activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics='binary_accuracy')

        model.fit(X, y ,epochs=100)
        
        return model


def main():
    inputfile = sys.argv[1]
    bfc = Build(inputfile)
    xtr,ytr = bfc.DataProcess()
    model = bfc.BuildModel(xtr,ytr)
    model.save('model')


if __name__ == "__main__":
     main()
