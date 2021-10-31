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
 
def list_imgs(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
#ココ↓
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

def main():
    # パラメータ
    batch_size = 5 # バッチサイズ
    num_classes = 6 # 分類クラス数(今回は15種類)
    epochs = 100     # エポック数(学習の繰り返し回数)
    dropout_rate = 0.2 # 過学習防止用：入力の20%を0にする（破棄）

    # データ格納用のディレクトリパス
    SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/law2/"
    SAVE_DATA_DIR_PATH1 = "img/"
    # ディレクトリがなければ作成
    os.makedirs(SAVE_DATA_DIR_PATH, exist_ok=True)

    # グラフ画像のサイズ
    FIG_SIZE_WIDTH = 12
    FIG_SIZE_HEIGHT = 10
    FIG_FONT_SIZE = 25

    data_x = []
    data_y = []
    num_classes = 6
# #親指
#     print("oya")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "oya"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(0) # 教師データ（正解）
    
#     print("oya_hitosashi")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "oya_hitosashi"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(1) # 教師データ（正解）

#     print("oya_naka")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "oya_naka"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(2) # 教師データ（正解）

#     print("oya_kusuri")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "oya_kusuri"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(3) # 教師データ（正解）

#     print("oya_ko")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "oya_ko"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(4) # 教師データ（正解）

#     print("oya_mu")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "oya_mu"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(5) # 教師データ（正解）
# #人差し指    
#     print("hitosashi")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "hitosashi"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(0) # 教師データ（正解）

#     print("hitosashi_oya")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "hitosashi_oya"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(1) # 教師データ（正解）

#     print("hitosashi_naka")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "hitosashi_naka"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(2) # 教師データ（正解）

#     print("hitosashi_kusuri")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "hitosashi_kusuri"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(3) # 教師データ（正解）

#     print("hitosashi_ko")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "hitosashi_ko"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(4) # 教師データ（正解）

#     print("hitosashi_mu")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "hitosashi_mu"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(5) # 教師データ（正解）

# #中指
#     print("naka")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "naka"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(0) # 教師データ（正解）

#     print("naka_oya")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "naka_oya"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(1) # 教師データ（正解）

#     print("naka_hitosashi")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "naka_hitosashi"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(2) # 教師データ（正解）

#     print("naka_kusuri")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "naka_kusuri"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(3) # 教師データ（正解）

#     print("naka_ko")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "naka_ko"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(4) # 教師データ（正解）

#     print("naka_mu")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "naka_mu"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(5) # 教師データ（正解）

# #薬指
#     print("kusuri")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "kusuri"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(0) # 教師データ（正解）

#     print("kusuri_oya")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "kusuri_oya"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(1) # 教師データ（正解）

#     print("kusuri_hitosashi")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "kusuri_hitosashi"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(2) # 教師データ（正解）

#     print("kusuri_naka")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "kusuri_naka"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(3) # 教師データ（正解）

#     print("kusuri_ko")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "kusuri_ko"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(4) # 教師データ（正解）

#     print("kusuri_mu")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "kusuri_mu"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(5) # 教師データ（正解）

# #小指
#     print("ko")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "ko"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(0) # 教師データ（正解）

#     print("ko_oya")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "ko_oya"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(1) # 教師データ（正解）

#     print("ko_hitosashi")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "ko_hitosashi"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(2) # 教師データ（正解）

#     print("ko_naka")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "ko_naka"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(3) # 教師データ（正解）

#     print("ko_kusuri")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "ko_kusuri"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(4) # 教師データ（正解）

#     print("ko_mu")
#     for filepath in list_csv(SAVE_DATA_DIR_PATH + "ko_mu"):
#         List=np.loadtxt(filepath, delimiter=',')
#         Listcut=np.array(List [:,0:8])
#         pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
#         img=img_to_array(pil_image)
#         data_x.append(img)
#         data_y.append(5) # 教師データ（正解）

#無動作
    print("mu")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "mu"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,0:8])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(0) # 教師データ（正解）

    print("mu_oya")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "mu_oya"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,0:8])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(1) # 教師データ（正解）

    print("mu_hitosashi")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "mu_hitosashi"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,0:8])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(2) # 教師データ（正解）

    print("mu_naka")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "mu_naka"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,0:8])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(3) # 教師データ（正解）

    print("mu_kusuri")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "mu_kusuri"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,0:8])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(4) # 教師データ（正解）

    print("mu_ko")
    for filepath in list_csv(SAVE_DATA_DIR_PATH + "mu_ko"):
        List=np.loadtxt(filepath, delimiter=',')
        Listcut=np.array(List [:,0:8])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)
        data_x.append(img)
        data_y.append(5) # 教師データ（正解）

# # クラス0の画像データ群をロード    
    # print("class0")
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "0"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(0) # 教師データ（正解）
    # print("class1")
    # # クラス1の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "1"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(1) # 教師データ（正解）
    # print("class2")
    # # クラス2の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "2"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(2) # 教師データ（正解）
    # print("class3")   
    #     # クラス3の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "3"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(3) # 教師データ（正解）
    # print("class4")    
    #     # クラス4の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "4"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(4) # 教師データ（正解）
    # print("class5")
    #     # クラス5の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "5"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(5) # 教師データ（正解）
    # print("class6")
    #     # クラス6の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "6"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(6) # 教師データ（正解）
    # print("class7")
    #     # クラス7の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "7"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(7) # 教師データ（正解）
    # print("class8")
    #     # クラス8の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "8"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(8) # 教師データ（正解）
    # print("class9")
    #     # クラス9の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "9"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(9) # 教師データ（正解）

    # print("class10")
    #     # クラス9の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "oya"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(10) # 教師データ（正解）
    # print("class11")
    #     # クラス9の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "hitosashi"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(11) # 教師データ（正解）
    # print("class12")
    #     # クラス9の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "naka"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(12) # 教師データ（正解）
    # print("class13")
    #     # クラス9の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "kusuri"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(13) # 教師データ（正解）
    # print("class14")
    #     # クラス9の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "ko"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(14) # 教師データ（正解）
    
    # print("class15")
    #     # クラス9の画像データ群をロード
    # for filepath in list_csv(SAVE_DATA_DIR_PATH + "mu"):
    #     List=np.loadtxt(filepath, delimiter=',')
    #     Listcut=np.array(List [:,0:8])
    #     pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
    #     img=img_to_array(pil_image)
    #     data_x.append(img)
    #     data_y.append(15) # 教師データ（正解）
#


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

    # 正解ラベルをone hotエンコーディング
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # データセットの個数を表示
    print(x_train.shape, 'x train samples')
    print(x_test.shape, 'x test samples')
    print(y_train.shape, 'y train samples')
    print(y_test.shape, 'y test samples')

  
    # モデルの構築
    model = Sequential()

    # 入力層:16×16*2
    # 【2次元畳み込み層】
    # Conv2D：2次元畳み込み層で、画像から特徴を抽出（活性化関数：relu）
    # 入力データにカーネルをかける（「2×2」の16種類のフィルタを各マスにかける）
    # 出力ユニット数：16（16枚分の出力データが得られる）
    
    model.add(Conv2D(16,(3,3), 
            padding='same', 
            input_shape=x_train.shape[1:],
            activation='relu'))
    model.add(Conv2D(16,(3,3), 
            padding='same', 
            input_shape=x_train.shape[1:],
            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(16,(3,3),
            padding='same',
            activation='relu'))
    model.add(Conv2D(16,(3,3),
            padding='same',
            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    # 平坦化（次元削減）
    # 1次元ベクトルに変換
    model.add(Flatten())

    # 全結合層
    # 出力ユニット数：512
    model.add(Dense(512, activation='relu'))
 


    # ドロップアウト(過学習防止用, dropout_rate=0.2なら20%のユニットを無効化）
    model.add(Dropout(dropout_rate))
    
    # 全結合層
    # 15分類（0から14まで）なので、ユニット数10, 分類問題なので活性化関数はsoftmax関数
    # Softmax関数で総和が1となるように、各出力の予測確率を計算
    model.add(Dense(num_classes, activation='softmax')) # 活性化関数：softmax

    # モデル構造の表示
    #model.summary()

    # コンパイル（多クラス分類問題）
    # 最適化：RMSpropを使用
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

 

    # 構築したモデルで学習（学習データ:trainのうち、10％を検証データ:validationとして使用）
    # verbose=1:標準出力にログを表示
    history = model.fit(x_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_split=0.15)

    # テスト用データセットで学習済分類器に入力し、パフォーマンスを計測
    score = model.evaluate(x_test, 
                            y_test,
                            verbose=0
                            )

    # パフォーマンス計測の結果を表示
    # 損失値（値が小さいほど良い）
    print('Test loss:', score[0])

    # 正答率（値が大きいほど良い）
    print('Test accuracy:', score[1])

    
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


    

    # # 学習過程をプロット
    # plot_history(history, 
    #             save_graph_img_path = SAVE_DATA_DIR_PATH + "graph.png", 
    #             fig_size_width = FIG_SIZE_WIDTH, 
    #             fig_size_height = FIG_SIZE_HEIGHT, 
    #             lim_font_size = FIG_FONT_SIZE)

    # モデル構造の保存
    open(SAVE_DATA_DIR_PATH  + "lawmodel.json","w").write(model.to_json())  

    # 学習済みの重みを保存
    model.save_weights(SAVE_DATA_DIR_PATH + "lawweight.h5")

    # 学習履歴を保存
    with open(SAVE_DATA_DIR_PATH + "lawhistory.json", 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    main()