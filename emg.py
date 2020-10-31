from __future__ import print_function
import collections
#モジュールmyoをインポート
import myo
import time
import sys
import numpy as np
from numpy.core.fromnumeric import reshape

#Emgクラス　サンプリング周波数200でデータを取得するクラス
class Emg(myo.DeviceListener):

  def __init__(self):
    print("class Emg instanced")
    self.rms = np.zeros((1,8))   
    self.result = np.zeros((1,8)) 
    self.add =  np.zeros((1,8)) 
    self.i = 0
    

  def on_connected(self, event):
    event.device.stream_emg(True)
 


  def on_emg(self,event):
    self.emg = np.array(event.emg)**2
    self.emg = np.reshape(self.emg,(1,8))
    
    if self.add.shape[0] <= 21:
      self.add = np.append(self.add,self.emg,axis=0)
    else:
      self.add = np.delete(self.add, 1, 0)

    sum = np.sum(self.add[1:],axis=0)
    ave = sum/20
    sqrt = np.sqrt(ave)
    np.set_printoptions(precision=0)
    print(sqrt)
      

  
  # def on_emg(self,event):
  #   self.emg = event.emg
  #   square = np.array(self.emg)**2
  #   self.rms += square
    

  #   if self.i%20 == 0:
  #     print(self.rms/20)
  #     self.i = 0
  #   self.i = self.i+1

    
    

  #   self.rms = np.zeros(8,dtype = int)   
  #   self.result = np.zeros(8,dtype = int)

#main関数
def main():
  myo.init(sdk_path=r'C:\work\myo-sdk-win-0.9.0-main')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  listener = Emg() #emgクラスのインスタンス

  try:
    start = time.time()
    while hub.run(listener.on_event, 500):
      current = time.time()
      t = float(current - start)
      if t >= 1:
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