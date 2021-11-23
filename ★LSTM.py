# -*- coding: utf-8 -*-
import numpy as np
from keras.optimizers import Adam, SGD
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.datasets import cifar10
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import os
import pickle
from PIL import Image

from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping
from keras import backend
 
def list_imgs(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
def list_csv(directory, ext='csv'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

def plot_history(history, 
                save_graph_img_path, 
                fig_size_width, 
                fig_size_height, 
                lim_font_size):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
   
    epochs = range(len(acc))

    # グラフ表示
    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = lim_font_size  # 全体のフォント
    #plt.subplot(121)

    # plot accuracy values
    plt.plot(epochs, acc, color = "blue", linestyle = "solid", label = 'train acc')
    plt.plot(epochs, val_acc, color = "green", linestyle = "solid", label= 'valid acc')
    #plt.title('Training and Validation acc')
    #plt.grid()
    #plt.legend()
 
    # plot loss values
    #plt.subplot(122)
    plt.plot(epochs, loss, color = "red", linestyle = "solid" ,label = 'train loss')
    plt.plot(epochs, val_loss, color = "orange", linestyle = "solid" , label= 'valid loss')
    #plt.title('Training and Validation loss')
    plt.legend()
    plt.grid()

    plt.savefig(save_graph_img_path)
    plt.close() # バッファ解放
    
def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

def main():


    # データ格納用のディレクトリパス
    # SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/law2/"
    SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/data/"

    SAVE_DATA_DIR_PATH1 = "img/"
    # ディレクトリがなければ作成
    os.makedirs(SAVE_DATA_DIR_PATH, exist_ok=True)



    data_x = []
    data_y = []
    num_classes = 6
# #親指
    print("oya")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "oya"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(0) # 教師データ（正解）
# #人差し指    
    print("hitosashi")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "hitosashi"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(1) # 教師データ（正解）
#中指
    print("naka")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "naka"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(2) # 教師データ（正解）
#薬指
    print("kusuri")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "kusuri"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(3) # 教師データ（正解）
#小指
    print("ko")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "ko"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(4) # 教師データ（正解）
#無動作
    print("mu")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "mu"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,:])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(5) # 教師データ（正解）
        
    # NumPy配列に変換
    data_x = np.asarray(data_x)

    # 学習データはNumPy配列に変換し
    data_y = np.asarray(data_y)
    


    # 学習用データとテストデータに分割 stratifyの引数でラベルごとの偏りをなくす
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.15,stratify=data_y)

    # 学習データはfloat32型に変換し、正規化(0～1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    x_train = x_train.reshape(x_train.shape[0], 100,4)
    x_test = x_test.reshape(x_test.shape[0], 100, 4)

    # 正解ラベルをone hotエンコーディング
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # データセットの個数を表示
    print(x_train.shape, 'x train samples')
    print(x_test.shape, 'x test samples')
    print(y_train.shape, 'y train samples')
    print(y_test.shape, 'y test samples')
    

    
    n_in = 4
    n_time = 100
    n_hidden = 128
    
    model = Sequential()
    model.add(Bidirectional(LSTM(n_hidden), input_shape=(n_time, n_in)))
    
    model.add(Dense(num_classes, kernel_initializer=weight_variable))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    model.summary()
    
    epochs = 100
    batch_size = 5
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    hist = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.15,
                    callbacks=[early_stopping])
    
    loss_and_metrics = model.evaluate(x_test, y_test)
    print(loss_and_metrics)
    
    #混同行列を表示
    from sklearn.metrics import confusion_matrix
    predict_classes = model.predict_classes(x_test)
    true_classes = np.argmax(y_test, 1)
    cmx = confusion_matrix(true_classes, predict_classes)
    print(cmx)
    #混同行列をヒートマップとして表示
    import seaborn as sns
    sns.heatmap(cmx, annot=True, fmt='g', square=True)
    plt.show()
    
    

  


if __name__ == '__main__':
    main()