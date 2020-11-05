#モジュールmyoをインポート
import myo
import time
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import collections
import serial
# from sklearn.neighbors import KNeighborsClassifier

#Emgクラス　サンプリング周波数200でデータを取得するクラス
class Emg(myo.DeviceListener):

  def __init__(self,mode):
    print("class Emg instanced mode=",mode)
    self.rms = np.zeros((1,8))   
    self.add =  np.zeros((1,8)) 
    self.element = np.zeros(1)
    self.mode = mode
    self.i = 0
    self.j = 0
    #_____knn_init________
    RMSList=np.loadtxt('RMSdata.csv', delimiter=',')
    element = RMSList [:,0:8]
    label = np.ravel(RMSList[:,8:9]) 
    rms_df = pd.DataFrame(element, columns=["e1","e2","e3","e4","e5","e6","e7","e8"])
    rms_target_data = pd.DataFrame(label, columns=["label"])
    self.knn = KNeighborsClassifier(n_neighbors=3)
    self.knn.fit(rms_df, rms_target_data)
    print("--------学習完了--------")
    #______serial_init_____
    # self.ser = serial.Serial('COM3',115200)

  def on_connected(self, event):
      event.device.stream_emg(True)

  def on_emg(self,event):
    self.emg = np.array(event.emg)**2

#_____________Mode0_Moving_RMS_________________________
    if self.mode == 0:
      self.emg = np.reshape(self.emg,(1,8))
      
      if self.add.shape[0] <= 21:
        self.add = np.append(self.add,self.emg,axis=0)
      else:
        self.add = np.delete(self.add, 1, 0)

      sum = np.sum(self.add[1:],axis=0)
      ave = sum/20
      sqrt = np.sqrt(ave)
      sqrt = np.round(sqrt, decimals=2)
      sqrt = np.array([sqrt])
      result = int(self.knn.predict(sqrt)[0])
      self.element = np.append(self.element,result)
      if self.i == 20:
        c = collections.Counter(self.element[1:])
        print(c.most_common()[0])
        self.element = np.zeros(1)
        self.i = 0
      self.i += 1

      # print(result)
      # result= str(result)
      # self.ser.write(bytes(result,'utf-8')) 
      
#_________________Mode1_RMS_____________________________
    elif self.mode == 1:
      self.rms += self.emg

      if self.j == 19:
        ave = self.rms/20
        sqrt = np.sqrt(ave)
        sqrt = np.round(sqrt, decimals=2)

        result = int(self.knn.predict(sqrt)[0])
        print(result)
        result= str(result)
        #arduino へ書き込み
        self.ser.write(bytes(result,'utf-8'))  

        self.rms = np.zeros((1,8))
        self.j = 0

      self.j += 1


#main関数
def main():
  myo.init(bin_path=r'./bin')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  listener = Emg(mode=0) #emgクラスのインスタンス (mode0 = Moving_RMS) (mode1 = RMS)
  try:
    start = time.time()
    while hub.run(listener.on_event,100):
      current = time.time()
      t = float(current - start)
      if t >= 20:
        print("stop"  ,t,"秒")
        # listener.ser.close()
        break

  except KeyboardInterrupt:
    # Ctrl-C を捕まえた！
    print('interrupted!')
    # なにか特別な後片付けが必要ならここに書く
    sys.exit(0)

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()