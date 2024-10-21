#@title try U_Bi_LSTM

from ast import main
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Conv1DTranspose, concatenate, BatchNormalization, Activation, add
from tensorflow.keras.layers import Bidirectional, LSTM, Conv1D, MaxPooling1D, Dropout, concatenate
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers import ELU, PReLU, LeakyReLU
from tensorflow.keras.utils import plot_model
import tensorflow as tf

###############
def Conv1dBlock(inputTensor, numFilters, kernelSize=3, doBatchNorm=True):
    # First Conv
    x = Conv1D(filters=numFilters, kernel_size=kernelSize,
               kernel_initializer='he_normal', padding='same')(inputTensor)
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second Conv
    x = Conv1D(filters=numFilters, kernel_size=kernelSize,
               kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def U_Bi_LSTM(inputImage, numFilters=16, droupouts=0.1, doBatchNorm=True):

    # Encoder Path

    c1 = Conv1dBlock(inputImage, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)
    # Add Bi-LSTM after first Conv block
    bl1 = Bidirectional(LSTM(numFilters * 1, return_sequences=True))(c1)
    bl1 = Conv1D(numFilters * 1, kernel_size=1, padding='same')(bl1)  # Adjust dimensions if needed
    p1 = MaxPooling1D((2))(bl1)
    p1 = Dropout(droupouts)(p1)

    c2 = Conv1dBlock(p1, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)
    # Add Bi-LSTM after second Conv block
    bl2 = Bidirectional(LSTM(numFilters * 2, return_sequences=True))(c2)
    bl2 = Conv1D(numFilters * 2, kernel_size=1, padding='same')(bl2)  # Adjust dimensions if needed
    p2 = MaxPooling1D((2))(bl2)
    p2 = Dropout(droupouts)(p2)

    c3 = Conv1dBlock(p2, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)
    # Add Bi-LSTM after third Conv block
    bl3 = Bidirectional(LSTM(numFilters * 4, return_sequences=True))(c3)
    bl3 = Conv1D(numFilters * 4, kernel_size=1, padding='same')(bl3)  # Adjust dimensions if needed
    p3 = MaxPooling1D((2))(bl3)
    p3 = Dropout(droupouts)(p3)

    c4 = Conv1dBlock(p3, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)
    # Add Bi-LSTM after fourth Conv block
    bl4 = Bidirectional(LSTM(numFilters * 8, return_sequences=True))(c4)
    bl4 = Conv1D(numFilters * 8, kernel_size=1, padding='same')(bl4)  # Adjust dimensions if needed
    p4 = MaxPooling1D((2))(bl4)
    p4 = Dropout(droupouts)(p4)

    # Bottleneck (final Conv block)
    c5 = Conv1dBlock(p4, numFilters * 16, kernelSize=3, doBatchNorm=doBatchNorm)

    # Decoder Path
    u6 = Conv1DTranspose(numFilters * 8, (3), strides=(2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(droupouts)(u6)
    c6 = Conv1dBlock(u6, numFilters * 8, kernelSize=3, doBatchNorm=doBatchNorm)

    u7 = Conv1DTranspose(numFilters * 4, (3), strides=(2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(droupouts)(u7)
    c7 = Conv1dBlock(u7, numFilters * 4, kernelSize=3, doBatchNorm=doBatchNorm)

    u8 = Conv1DTranspose(numFilters * 2, (3), strides=(2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(droupouts)(u8)
    c8 = Conv1dBlock(u8, numFilters * 2, kernelSize=3, doBatchNorm=doBatchNorm)

    u9 = Conv1DTranspose(numFilters * 1, (3), strides=(2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(droupouts)(u9)
    c9 = Conv1dBlock(u9, numFilters * 1, kernelSize=3, doBatchNorm=doBatchNorm)

    output = Conv1D(1, (1), activation=None)(c9)
    model = Model(inputs=[inputImage], outputs=[output])
    return model

if __name__ == "__main__":
    inputs = tf.keras.layers.Input((2400, 1))
    model = U_Bi_LSTM(inputs, droupouts=0.07)
    #model.summary()
