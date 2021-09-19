# import os
# import re
# import numpy as np
# from keras.preprocessing.image import array_to_img,img_to_array,load_img
# from PIL import Image

# def list_csv(directory, ext='csv'):
#     return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]



# # 入力画像のパラメータ
# img_width = 8 # 入力画像の幅
# img_height = 100 # 入力画像の高さ
# img_ch = 1 # グレースケールにて学習

# # データ格納用のディレクトリパス
# SAVE_DATA_DIR_PATH = "csv/"

# # ディレクトリがなければ作成
# os.makedirs(SAVE_DATA_DIR_PATH, exist_ok=True)

# # グラフ画像のサイズ
# FIG_SIZE_WIDTH = 12
# FIG_SIZE_HEIGHT = 10
# FIG_FONT_SIZE = 25

# data_x = []
# data_y = []
# num_classes = 15

# print(list_csv(SAVE_DATA_DIR_PATH + "0"))

# List=np.loadtxt('csv/0/MRMSdata1.csv', delimiter=',')
# Listcut=np.array(List [:,0:8])

# print(Listcut)

# pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))

# print(pil_image)

# img=img_to_array(pil_image)
# print(img)

# t=load_img("./img/0/Limg1.jpg",color_mode='grayscale',target_size=(img_width,img_height))

# print(t)
# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import time


def main():
    # 入力画像のパラメータ
    img_width = 32 # 入力画像の幅
    img_height = 32 # 入力画像の高さ
    img_ch = 3 # 3ch画像（RGB）

    # 入力データ数
    num_data = 1

    # データの保存先(自分の環境に応じて適宜変更)
    SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/csv2/"

    # ラベル
    labels =['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'oya', 'hitosashi', 'naka', 'kusuri', 'ko','mu']

    # 保存したモデル構造の読み込み
    model = model_from_json(open(SAVE_DATA_DIR_PATH + "model.json", 'r').read())
    print("model_read_ok")
    # 保存した学習済みの重みを読み込み
    model.load_weights(SAVE_DATA_DIR_PATH + "weight.h5")
    print("weights_read_ok")
    # 画像の読み込み（32×32にリサイズ）
    # 正規化, 4次元配列に変換（モデルの入力が4次元なので合わせる）
    time.sleep(1)
    for i in labels:
        start = time.time()

        List=np.loadtxt('csv/'+str(i)+'/MRMSdata20.csv', delimiter=',')
        Listcut=np.array(List [:,0:8])
        pil_image = Image.fromarray(np.rot90(np.uint8(Listcut)))
        img=img_to_array(pil_image)

        img = img.astype('float32')/255.0
        img = np.array([img])

    # y_pred = model.predict(img)
        y_pred = model.predict(img)
        t = time.time()-start
        number_pred = np.argmax(y_pred) 
        print("入力画像番号",i,"|","処理時間",t,"|","認識結果", number_pred)

    

    # 最も確率の高い要素番号
    number_pred = np.argmax(y_pred) 

    # 予測結果の表示

    print("y_pred:", y_pred)  # 出力値
    print("number_pred:", number_pred)  # 最も確率の高い要素番号
    print('label_pred：', labels[int(number_pred)]) # 予想ラベル（最も確率の高い要素）


    """
    predict_y: [[1.2638741e-20 4.6908645e-21 1.0000000e+00]]
    predict_number: 2
    predict_label： マグカップ
    """


if __name__ == '__main__':
    main()