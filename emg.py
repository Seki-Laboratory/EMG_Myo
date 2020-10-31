from __future__ import print_function
import collections
#モジュールmyoをインポート
import myo
import time
import sys
import numpy as np
from numpy.core.fromnumeric import reshape
import csv

#Emgクラス　サンプリング周波数200でデータを取得するクラス
class Emg(myo.DeviceListener):

  def __init__(self,mode):
    print("class Emg instanced mode=",mode)
    self.rms = np.zeros((1,8))   
    self.add =  np.zeros((1,8)) 
    self.mode = mode
    self.i = 0
    self.j = 0
    

  def on_connected(self, event):
      event.device.stream_emg(True)

  def on_emg(self,event):
    self.emg = np.array(event.emg)**2
#______________Moving_RMS_________________________
    if self.mode == 0:
      self.emg = np.array(event.emg)**2
      self.emg = np.reshape(self.emg,(1,8))
      
      if self.add.shape[0] <= 21:
        self.add = np.append(self.add,self.emg,axis=0)
      else:
        self.add = np.delete(self.add, 1, 0)

      sum = np.sum(self.add[1:],axis=0)
      ave = sum/20
      sqrt = np.sqrt(ave)
      sqrt = np.round(sqrt, decimals=2)
      print(sqrt)

      if self.i <= 19:
        with open('MRMSdata.csv', 'a') as f:
          writer = csv.writer(f, lineterminator='\n') # 行末は改行
          writer.writerow(sqrt)
        self.i += 1

#___________________RMS_____________________________
    elif self.mode == 1:
      self.rms += self.emg

      if self.j == 19:
        ave = self.rms/20
        sqrt = np.sqrt(ave)
        sqrt = np.round(sqrt, decimals=2)
        sqrt = np.reshape(sqrt,(8))
        print(sqrt)

        if self.i <= 19:
          with open('RMSdata.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n') # 行末は改行
            writer.writerow(sqrt)
          self.i += 1

        self.rms = np.zeros((1,8))
        self.j = 0

      self.j += 1

#main関数
def main():
  myo.init(sdk_path=r'C:\work\myo-sdk-win-0.9.0-main')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  listener = Emg(mode=1) #emgクラスのインスタンス

  try:
    start = time.time()
    while hub.run(listener.on_event, 100):
      current = time.time()
      t = float(current - start)
      if listener.i >= 20:
        print("stop"  ,t,"秒")
        break

  except KeyboardInterrupt:
    # Ctrl-C を捕まえた！
    print('interrupted!')
    # なにか特別な後片付けが必要ならここに書く
    sys.exit(0)

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()