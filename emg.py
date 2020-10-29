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

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    self.emg = event.emg
    #print(self.emg,len(self.emg))
    sq = np.array(self.square())
    b=np.ones((20,8))/20
    #y2=np.convolve(sq, b, mode='vaild')#移動平均
    print(b)

  
  def square(self):
    square = np.array(self.emg)**2
    return square

      
      
#main関数
def main():
  myo.init(sdk_path=r'C:\work\myo-sdk-win-0.9.0-main')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  listener = Emg() #emgクラスのインスタンス
  try:
    while hub.run(listener.on_event, 500):
      pass
  except KeyboardInterrupt:
    # Ctrl-C を捕まえた！
    print('interrupted!')
    # なにか特別な後片付けが必要ならここに書く
    sys.exit(0)

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()