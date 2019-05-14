import os
from keras.optimizers import *
from keras.models import load_model, Sequential
from keras.layers import *
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self):
        """
        Base class for classification models. It provides general abstract
        methods required for applying a machine learning techniques.
        """
        self.model = None
        self.optimizer = None
        self.activation = None

    def compile(self, optimizer, loss='mean_squared_error'):
        """
        Compile model with given configuration.

        :param optimizer: Instance of optimizer.
        :param loss: String (name of loss function) or custom function.
        """
        pass

    def save_model(self, model_name, path):
        """
        Saves a model into a H5py file

        :param model_name: File name.
        :param path: Path of the folder.
        """
        pass

    def load_model(self, model_name, path):
        """
        Saves a model into a H5py file

        :param model_name: File name.
        :param path: Path of the folder.
        """
        pass

    def train(self, x_train, y_train, epochs=100, batch_size=32,
              validation_split=0.1):
        """
        Trains the model on the training data with given settings.

        :param x_train:  Numpy array of training data.
        :param y_train:  Numpy array of target data.
        :param epochs: Integer. Number of epochs during training.
        :param batch_size: Number of samples per gradient update.
        :param validation_split: Float between 0 and 1. Fraction of the
        training data to be used as validation data.
        """
        pass

    def predict(self, x):
        """
        Predicts the output of the model for the given data as input.

        :param x: Input data as Numpy arrays.
        """
        pass

    def plot_history(self):
        """
        Plots loss vs epoch graph.
        """
        plt.plot(self.history.history['loss'])
        if 'val_loss' in self.history.history.keys():
            plt.plot(self.history.history['val_loss'])
        plt.title('Autoencoder loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

    def reset(self):
        self.model.reset_states()


class XRayBinaryClassifier(Classifier):
    def __init__(self, activation='relu',
                 optimizer=Adam(lr=0.0001, decay=1e-6)):
        """
        Feature classification model for X-Ray Binaries.

        :param activation: String (activation function name).
        :param optimizer: Instance of optimizer.
        """
        self.activation = activation
        self.optimizer = optimizer

        model = Sequential()
        model.add(Dense(32, input_shape=(3,), activation=self.activation))
        model.add(Dense(32, activation=self.activation))
        model.add(Dense(16, activation=self.activation))
        model.add(Dense(3, activation='softmax'))
        self.model = model

    def compile(self, loss='mean_squared_error'):
        """
        Compile model with given configuration.

        :param loss: String (name of loss function) or custom function.
        """
        self.model.compile(self.optimizer,
                           loss=loss, metrics=['accuracy'])

    def save_model(self, model_name, path='models/'):
        """
        Saves a model into a H5py file.
        :param model_name: File name.
        :param path: Pa
        """
        path = 'models/' + model_name
        self.model.save(path)

    def load_model(self, model_name, path='models/'):
        """
        Saves a model into a H5py file

        :param model_name: File name.
        :param path: Path of the folder.
        """
        path = 'models/' + model_name
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError("Model does not exists")

    def train(self, x_train, y_train, epochs=100, batch_size=32,
              validation_split=0.1):
        """
        Trains the model on the training data with given settings.

        :param x_train:  Numpy array of training data.
        :param y_train:  Numpy array of target data.
        :param epochs: Integer. Number of epochs during training.
        :param batch_size: Number of samples per gradient update.
        :param validation_split: Float between 0 and 1. Fraction of the
        training data to be used as validation data.
        """
        if not isinstance(x_train, np.ndarray) and\
                isinstance(y_train, np.ndarray):
            raise ValueError('Input array should be numpy arrays')

        self.model.fit(x_train, y_train, epochs=epochs, shuffle=True,
                       batch_size=batch_size,
                       validation_split=validation_split)

    def test(self, x_test):
        return self.model.predict_classes(x_test)


class AtlasVarStarClassifier(Classifier):
    def __init__(self, activation='relu', optimizer=Adam(lr=0.01, decay=0.01),
                 input_size=22, num_classes=9):
        self.activation = activation
        self.optimizer = optimizer
        model = Sequential()
        model.add(Dense(64, input_shape=(input_size,),
                        activation=self.activation))
        model.add(Dense(64, activation=self.activation))
        model.add(Dense(32, activation=self.activation))
        model.add(Dense(16, activation=self.activation))
        model.add(Dense(num_classes, activation='softmax'))
        self.model = model

    def compile(self, loss='mean_squared_error'):
        """
        build the model
        """
        self.model.compile(self.optimizer,
                           loss=loss, metrics=['accuracy'])

    def save_model(self, model_name, path='models/'):
        """
        save model
        """
        path = 'models/' + model_name
        self.model.save(path)

    def load_model(self, model_name, path='models/'):
        """
        load saved model
        """
        path = 'models/' + model_name
        if os.path.exists(path):
            self.model = load_model(path)
        else:
            raise FileNotFoundError("Model does not exists")

    def train(self, x_train, y_train, epochs=100, batch_size=32,
              validation_split=0.1):

        if not isinstance(x_train, np.ndarray) and\
                isinstance(y_train, np.ndarray):
            raise ValueError('Input array should be numpy arrays')

        self.model.fit(x_train, y_train, epochs=epochs, shuffle=True,
                       batch_size=batch_size,
                       validation_split=validation_split)

    def test(self, x_test):
        return self.model.predict_classes(x_test)


class HTRU1Classifier(Classifier):
    def __init__(self, input_dim, activation='relu', padding='same',
                 dropout=0.25, num_classes=2):
        """
        CNN Classification of pulsars and non-pulsars data released by HTRU
        survey as Data Release 1. The dataset has same structure as CIFAR-10
        dataset.

        :param input_dim: Set. Dimension of input data.
        :param activation: String. Activation function name.
        :param padding: Sting. Padding type.
        :param dropout: Float between 0 and 1. Dropout value.
        :param num_classes: Integer. Number of classes.
        """
        self.input_dim = input_dim
        self.activation = activation
        self.padding = padding
        self.history = None

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding=padding,
                              input_shape=input_dim))
        self.model.add(Activation(activation))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation(activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(dropout))

        self.model.add(Conv2D(64, (3, 3), padding=padding))
        self.model.add(Activation(activation))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation(activation))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(dropout))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation(activation))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

    def compile(self, optimizer, loss='categorical_crossentropy'):
        """
        Compile model with given configuration.

        :param optimizer: Instance of optimizer.
        :param loss: String (name of loss function) or custom function.
        """
        self.model.compile(loss=loss, optimizer=optimizer)

    def train(self, x_train, y_train, epochs=100, batch_size=32,
              reset_weights=True, class_weight=None, validation_data=None,
              verbose=1):
        """
        Trains the model on the training data with given settings.

        :param x_train:  Numpy array of training data.
        :param y_train:  Numpy array of target data.
        :param epochs: Integer. Number of epochs during training.
        :param batch_size: Number of samples per gradient update.
        :param reset_weights: Bool. Whether to reset weights of trained model.
        :param class_weight: Dictionary of weight assigned to each class for
        weighted loss function.
        :param validation_data: Numpy array of validation data.
        :param verbose: Value 0, 1, or 2.
        """
        if reset_weights:
            self.reset()

        self.history = self.model.fit(x_train, y_train, batch_size=batch_size,
                                      epochs=epochs,
                                      validation_data=validation_data,
                                      class_weight=class_weight, shuffle=True,
                                      verbose=verbose)

    def predict(self, x):
        """
        Predicts the output of the model for the given data as input.

        :param x: Input data as Numpy arrays.
        """
        return self.model.predict_classes(x)
