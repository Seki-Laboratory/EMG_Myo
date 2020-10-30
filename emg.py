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
    self.rms = np.zeros(8,dtype = int)   
    self.result = np.zeros(8,dtype = int)  
    

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self,event):
    self.emg = event.emg
    square = np.array(self.emg)**2

    for i in range(20):
      self.rms += square
      print(i+1)

    
    

    self.rms = np.zeros(8,dtype = int)   
    self.result = np.zeros(8,dtype = int)

#main関数
def main():
  myo.init(sdk_path=r'C:\work\myo-sdk-win-0.9.0-main')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  listener = Emg() #emgクラスのインスタンス
  start = time.time()
  try:
    while hub.run(listener.on_event, 500):
      finish = time.time()
      t = finish - start
      if t >= 2:
        finish = time.time()
        print("stop",finish-start,"秒")
        break

  except KeyboardInterrupt:
    # Ctrl-C を捕まえた！
    print('interrupted!')
    # なにか特別な後片付けが必要ならここに書く
    sys.exit(0)

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()