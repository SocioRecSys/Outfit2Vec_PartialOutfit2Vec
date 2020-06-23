##### Style Classification Experiment ######

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras.utils.np_utils import to_categorical
import os

import keras.backend as K

### CNN - Style Classification Experiment
class StyleClassification_FNN(object):

    def precision(self,y_true, y_pred):
         # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if c3 == 0:
            return 0

        # How many selected items are relevant?
        precision = c1 / c2

        return precision

    def recall(self,y_true, y_pred):
        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if c3 == 0:
            return 0

        # How many relevant items are selected?
        recall = c1 / c3

        return recall

    def f1_score(self, y_true, y_pred):

        # Count positive samples.
        c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

        # If there are no true samples, fix the F1 score at 0.
        if c3 == 0:
            return 0

        # How many selected items are relevant?
        precision = c1 / c2

        # How many relevant items are selected?
        recall = c1 / c3

        # Calculate f1_score
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def __init__(self, dataPath):
        self.testSize = 0.3
        self.max_words = 1000
        self.batch_size = 32
        self.epochs = 20
        self.dataPath = dataPath

    def processModel(self):

        df = pd.read_csv(self.dataPath, names={'col1', 'col2'})

        train_size = int(len(df) * (1-self.testSize))

        train_outfits = df['col2'][:train_size]
        train_tags = df['col1'][:train_size]

        test_outfits = df['col2'][train_size:]
        test_tags = df['col1'][train_size:]

        tokenize = text.Tokenizer(nb_words=self.max_words, char_level=False)
        tokenize.fit_on_texts(train_outfits)

        x_train = tokenize.texts_to_matrix(train_outfits)
        x_test = tokenize.texts_to_matrix(test_outfits)

        encoder = LabelEncoder()
        encoder.fit(train_tags)
        y_train = encoder.transform(train_tags)
        y_test = encoder.transform(test_tags)

        num_classes = np.max(y_train) + 1
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        # Build the model
        model = Sequential()
        model.add(Dense(512, input_shape=(self.max_words,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', #the optimizer was Adam
                      metrics=['accuracy', self.precision, self.recall, self.f1_score])


        model.fit(x_train, y_train,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            validation_split=0.3)

        score = model.evaluate(x_test, y_test,
                               batch_size=self.batch_size, verbose=1)
        print('Test accuracy:', score[1])
        print('Score metrics are:', score)
        print("document # " + self.dataPath)

# Iterate over folder of data files that are used in the evaluation
for filename in os.listdir('REMOVED'):
    obj = StyleClassification_FNN(filename)
    obj.processModel()
