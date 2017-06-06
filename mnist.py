from theano.sandbox import cuda
cuda.use('gpu2')
%matplotlib inline
import utils
from utils import *
from __future__ import division, print_function
from keras.datasets import mnist
from keras import backend
backend.set_image_dim_ordering('th')



#data normalization



def norm_input(x): 
    return (x-mean_px)/std_px

def get_lin_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28)),
        Flatten(),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_model_bn():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,28,28)),
        Convolution2D(32,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(32,3,3, activation='relu'),
        MaxPooling2D(),
        BatchNormalization(axis=1),
        Convolution2D(64,3,3, activation='relu'),
        BatchNormalization(axis=1),
        Convolution2D(64,3,3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#Ensembling 

def fit_model():
    model = get_model_bn_do()
    model.fit_generator(batches, batches.n, nb_epoch=1, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.n)
    model.optimizer.lr=0.1
    model.fit_generator(batches, batches.n, nb_epoch=4, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.n)
    model.optimizer.lr=0.01
    model.fit_generator(batches, batches.n, nb_epoch=12, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.n)
    model.optimizer.lr=0.001
    model.fit_generator(batches, batches.n, nb_epoch=18, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.n)
    return model

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    mean_px = X_train.mean().astype(np.float32)
    
    std_px = X_train.std().astype(np.float32)
    #expand is i think for imagedatagenerator, which for grey scale is expecting 1 for the 1st axis.
    X_test = np.expand_dims(X_test,1)
    X_train = np.expand_dims(X_train,1)

    y_train = onehot(y_train)
    y_test = onehot(y_test)

    
    gen = image.ImageDataGenerator()
    batches = gen.flow(X_train, y_train)

    test_batches = gen.flow(X_test, y_test, batch_size=64)

    
    model = get_model_bn()
    
    model.fit_generator(batches, batches.n, nb_epoch=1, 
                        validation_data=test_batches, nb_val_samples=test_batches.n)
    
    model.optimizer.lr=0.001
    model.fit_generator(batches, batches.n, nb_epoch=12, 
                    validation_data=test_batches, nb_val_samples=test_batches.n)
    
    #ENSEMBLING
    models = [fit_model() for i in range(6)]
    
    path = "data/mnist/"
    model_path = path + 'models/'
    for i,m in enumerate(models):
        m.save_weights(model_path+'cnn-mnist23-'+str(i)+'.pkl')
    evals = np.array([m.evaluate(X_test, y_test, batch_size=256) for m in models])
    evals.mean(axis=0)

    all_preds = np.stack([m.predict(X_test, batch_size=256) for m in models])

    avg_preds = all_preds.mean(axis=0)
    keras.metrics.categorical_accuracy(y_test, avg_preds).eval()
    
    