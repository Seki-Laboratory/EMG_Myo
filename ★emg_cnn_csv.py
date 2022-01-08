#モジュールmyoをインポート
import myo
import time
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import collections
import cv2

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import serial
from PIL import Image

# from sklearn.neighbors import KNeighborsClassifier

#Emgクラス　サンプリング周波数200でデータを取得するクラス
class Emg(myo.DeviceListener):

  def __init__(self,mode):
    print("class Emg instanced mode=",mode)
    self.rms = np.zeros((1,8))   
    self.add =  np.zeros((1,8)) 
    self.rms_add = np.zeros((1,8)) 
    self.element = np.zeros(1)
    self.mode = mode
    self.i = 0
    self.j = 0
    self.list = 0
    self.n =0
    self.geta127 = [128,128,128,128,128,128,128,128]
    #_____cnn_init________
    # データの保存先(自分の環境に応じて適宜変更)
    SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/demo/"
    # ラベル
    print("ok")
    self.labels =['oya', 'hitosashi', 'naka', 'kusuri', 'ko',"mu"]
    # self.labels =['oya', 'hitosashi', 'naka', 'kusuri', 'ko',"mu"]
    # 保存したモデル構造の読み込み
    self.model = model_from_json(open(SAVE_DATA_DIR_PATH+"lawmodel.json", 'r').read())
    print("ok")
    # 保存した学習済みの重みを読み込み
    self.model.load_weights(SAVE_DATA_DIR_PATH + "lawweight.h5")
    print("モデルと重みの読み込み完了")
    #______serial_init_____
    self.ser = serial.Serial('COM5',9600)

    

  

  def on_connected(self, event):
      event.device.stream_emg(True)

  def on_emg(self,event):
    start = time.time()

    
#_____________Mode0_Moving_RMS_________________________
    if self.mode == 0:
      self.emg = np.array(event.emg)**2
      self.emg = np.reshape(self.emg,(1,8))
#_____________rms_calc      
      if self.add.shape[0] <= 21:
        self.add = np.append(self.add,self.emg,axis=0)
      else:
        self.add = np.delete(self.add, 1, 0)
        sum = np.sum(self.add[1:],axis=0)
        ave = sum/20
        sqrt = np.sqrt(ave)
        sqrt = np.round(sqrt, decimals=2)
        sqrt = np.array([sqrt])
#_____________rms_list    
        if self.rms_add.shape[0] <= 101:
          self.rms_add = np.append(self.rms_add,sqrt,axis=0)
        else:
          self.rms_add = np.delete(self.rms_add, 1, 0)
          rms_list = np.round(self.rms_add[1:],decimals=2)

          pil_image = Image.fromarray(np.rot90(np.uint8(rms_list)))
          img=img_to_array(pil_image)
          img = img.astype('float32')/255.0
          img = np.array([img])
  # 分類機に入力データを与えて予測（出力：各クラスの予想確率）
          y_pred = self.model.predict(img)
  # 最も確率の高い要素番号
          number_pred = np.argmax(y_pred) 
          print("認識結果",self.labels[int(number_pred)])

  # #読み飛ばし  
  #         if 0 == self.n%2:
  #     # 分類機に入力データを与えて予測（出力：各クラスの予想確率）
  #             y_pred = self.model.predict(img)
  #             t = time.time()-start
  #     # 最も確率の高い要素番号
  #             number_pred = np.argmax(y_pred) 
  #             print("認識結果",self.labels[int(number_pred)])
  #             self.n = 0
  #         self.n = self.n+1

        # self.element = np.append(self.element,result)
        # if len(self.element) == 16:
        #   c = collections.Counter(self.element[1:])
        #   # print(c.most_common()[0])
        #   list = c.most_common()[0]
        #   if list[1] == 15:
        #     print(list)
        #     list_c = int(list[0])
        #   else:
        #     print("none")
        #     pass
        #   self.element = np.delete(self.element,1)

#______________Mode1_Law_EMG
    if self.mode == 1:
      self.emg = np.array(event.emg)
      gemg = self.emg+self.geta127
      gemg = np.reshape(gemg,(1,8))

      if self.rms_add.shape[0] <= 101:
        self.rms_add = np.append(self.rms_add,gemg,axis=0)
      else:
        self.rms_add = np.delete(self.rms_add, 1, 0)
        rms_list = np.round(self.rms_add[1:],decimals=2)

        pil_image = Image.fromarray(np.rot90(np.uint8(rms_list)))
        img=img_to_array(pil_image)
        img = img.astype('float32')/255.0
        img = np.array([img])
# 分類機に入力データを与えて予測（出力：各クラスの予想確率）
        y_pred = self.model.predict(img)
# 最も確率の高い要素番号
        number_pred = np.argmax(y_pred) 
        # print("認識結果",self.labels[int(number_pred)])
        # self.ser.write(bytes(str(number_pred),'utf-8')) 
        self.element = np.append(self.element,number_pred)
        if len(self.element) == 15:
          c = collections.Counter(self.element[1:])
          # print(c.most_common()[0])
          list = c.most_common()[0]
          if list[1] == 14:
            list_c = int(list[0])
            print("認識結果",self.labels[list_c])
            self.ser.write(bytes(str(list_c),'utf-8')) 
          else:
            print("認識結果","none")
            pass
          self.element = np.delete(self.element,1)




#main関数
def main():
  myo.init(bin_path=r'./bin')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  listener = Emg(mode=1) #emgクラスのインスタンス (mode0 = Moving_RMS) (mode1 = law_EMG)
  try:
    start = time.time()
    while hub.run(listener.on_event,5):
      current = time.time()
      t = float(current - start)
      # if t >= 20:
      #   print("stop"  ,t,"秒")
      #   # listener.ser.close()
      #   break

  except KeyboardInterrupt:
    # Ctrl-C を捕まえた！
    print('interrupted!')
    # なにか特別な後片付けが必要ならここに書く
    sys.exit(0)

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()