from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Input, Add
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
import time
from zipfile import ZipFile
import pickle
import tempfile
import matplotlib.pyplot as plt
import argparse
import tensorflow.keras.backend as K

class ResNetCIFAR:
    
    def __init__(self):
        self.model = None
        self.loss = []
        self.acc = []
        self.eval_loss = []
        self.eval_acc = []
        
    def conv_block(self, inputs, filters, kernel_size=(3,3), strides=(1,1), name=None):
        conv_layer_1 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(inputs)
        bn_1 = BatchNormalization()(conv_layer_1)
        activation_1 = Activation('relu')(bn_1)
        
        return activation_1
    
    def residual_block(self, inputs, filters, kernel_size=(3,3), strides=(1,1)):
        
        if strides == (1,1):
            x_shortcut = inputs
        else:
            x_shortcut = Conv2D(filters, kernel_size=(1,1), strides=(2,2), padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(inputs)
            
        conv_layer_1 = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(inputs)
        bn_1 = BatchNormalization()(conv_layer_1)
        activation_1 = Activation('relu')(bn_1)
        
        conv_layer_2 = Conv2D(filters, kernel_size=kernel_size, strides=(1,1), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))(activation_1)
        bn_2 = BatchNormalization()(conv_layer_2)
        
        merge = Add()([bn_2, x_shortcut])
        activation_2 = Activation('relu')(merge)
        
        return activation_2
        
    def create_model(self, N):
        # N in {3, 5, 7, 9, 18, 200}
        
        inputs = Input(shape=(32,32,3))
        
        x = self.conv_block(inputs, 16, kernel_size=(3,3))
        
        for i in range(N):
            x = self.residual_block(x, 16)
            
        x = self.residual_block(x, 32, strides=(2,2))
        for i in range(N-1):
            x = self.residual_block(x, 32)
        
        x = self.residual_block(x, 64, strides=(2,2))
        for i in range(N-1):
            x = self.residual_block(x, 64)
        
        x = GlobalAveragePooling2D()(x)
        
        # White
        x = Dense(10, activation='softmax', kernel_regularizer=l2(0.0001), kernel_initializer='he_normal')(x)
        
        model = Model(inputs=inputs, outputs=x)
        sgd = SGD(lr=.1, momentum=.9)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        
    def train_model(self, train_batches, valid_batches, num_iterations=64000, init_iter=1, log_frequency=10):
        train_time = 0
        total_train_time = 0
        eval_time = 0
        
        try:
            for iters in range(init_iter, num_iterations+1):
                # lr_decay
                if iters == 32000:
                    K.set_value(self.model.optimizer.lr, 0.01)
                elif iters == 48000:
                    K.set_value(self.model.optimizer.lr, 0.001)
                    # print(self.model.optimizer.get_config())

                train_time = time.time()
                imgs, labels = next(train_batches)
                loss, acc = self.model.train_on_batch(imgs, labels, reset_metrics=False)
                self.loss.append((iters, loss))
                self.acc.append((iters, acc))

                total_train_time += time.time() - train_time
                if iters % log_frequency == 0:
                    print(f'iteration : {iters}, loss : {loss:.4f}, acc : {acc:.4f} (time/iter : {total_train_time/log_frequency:.4f}s, total_time : {total_train_time:.4f}s)')
                    total_train_time = 0

                # Evaluate on eval dataset
                if iters % 100 == 0:
                    eval_time = time.time()
                    eval_loss, eval_acc = self.model.evaluate_generator(valid_batches)
                    self.eval_loss.append((iters, eval_loss))
                    self.eval_acc.append((iters, eval_acc))
                    if iters % (log_frequency*100) == 0:
                        print(f'>>> eval_loss : {eval_loss:.4f}, eval_acc : {eval_acc:.4f} (eval_time : {time.time() - eval_time:.4f}s)')
        except KeyboardInterrupt:
            print(f'iteration : {iters}, loss : {loss:.4f}, acc : {acc:.4f}')
                    
    def save_model(self, filename):
        with ZipFile(filename, 'w') as zipfile:
            with tempfile.TemporaryDirectory() as temp_dir:

                self.model.save(temp_dir + '/model.h5')
                zipfile.write(temp_dir + '/model.h5', arcname='model.h5')

                with open(temp_dir + '/metrics.pickle', 'wb') as pickle_file:
                    pickle.dump(self.loss, pickle_file)
                    pickle.dump(self.acc, pickle_file)
                    pickle.dump(self.eval_loss, pickle_file)
                    pickle.dump(self.eval_acc, pickle_file)
                zipfile.write(temp_dir + '/metrics.pickle', arcname='metrics.pickle')

    @staticmethod
    def load_model(filename):
        with ZipFile(filename) as zipfile:
            with tempfile.TemporaryDirectory() as temp_dir:
                zipfile.extractall(temp_dir)

                resnet_cifar_loaded = ResNetCIFAR()
                resnet_cifar_loaded.model = load_model(temp_dir + '/model.h5')

                with open(temp_dir + '/metrics.pickle', 'rb') as pickle_file:
                    resnet_cifar_loaded.loss = pickle.load(pickle_file)
                    resnet_cifar_loaded.acc = pickle.load(pickle_file)
                    resnet_cifar_loaded.eval_loss = pickle.load(pickle_file)
                    resnet_cifar_loaded.eval_acc = pickle.load(pickle_file)
    
        return resnet_cifar_loaded
    
    def plot_loss_and_accuracy(self, step=1, start=0):
        eval_step = 1 if step/100<0 else int(step/100)
        
        # Loss
        f1 = plt.figure(1)
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.plot(*zip(*self.loss[start::step]), label='train_loss')
        plt.plot(*zip(*self.eval_loss[int(start/100)::eval_step]), label='eval_loss')
        plt.legend()

        # Accuracy
        f2 = plt.figure(2)
        plt.xlabel('iter')
        plt.ylabel('accuracy')
        plt.plot(*zip(*self.acc[start::step]), label='acc')
        plt.plot(*zip(*self.eval_acc[int(start/100)::eval_step]), label='eval_acc')
        plt.legend()
        return self.acc[start::step]
    