#モジュールmyoをインポート
import myo
import time
import sys
import numpy as np
import csv
from msvcrt import getch

#Emgクラス　サンプリング周波数200でデータを取得するクラス
class Emg(myo.DeviceListener):

  def __init__(self,mode):
    print("class Emg instanced mode=",mode)
    self.rms = np.zeros((1,8))   
    self.add =  np.zeros((1,8)) 
    self.mode = mode
    self.i = 0
    self.j = 0
    self.label = int(0)
    self.stop = 0
  
  def on_connected(self, event):
      event.device.stream_emg(True)
      print("stream_start")

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
      print(sqrt)

      if self.i <= 19:
        with open('MRMSdata.csv', 'a') as f:
          writer = csv.writer(f, lineterminator='\n') # 行末は改行
          writer.writerow(sqrt)
        self.i += 1

#_________________Mode1_RMS_____________________________
    elif self.mode == 1:

      self.rms += self.emg

      if self.j == 19:
        ave = self.rms/20
        sqrt = np.sqrt(ave)
        sqrt = np.round(sqrt, decimals=2)
        sqrt = np.reshape(sqrt,(8))
        sqrt = np.append(sqrt,self.label)
        print(sqrt)

        if self.i <= 19:
          with open('RMSdata.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n') # 行末は改行
            writer.writerow(sqrt)
          self.i += 1
        elif self.i >= 20:
          event.device.stream_emg(False)
          print("学習データを取得しました。  [続行 = Enter][終了 = Esc] ")
          while True:
            key = ord(getch())
            if key == 13:
              self.label += 1
              event.device.stream_emg(True)
              self.i = 0
              break
            elif key == 27:
              self.stop = 1
              break

        self.rms = np.zeros((1,8))
        self.j = 0

      self.j += 1

#main関数
def main():
  myo.init(sdk_path=r'C:\work\myo-sdk-win-0.9.0-main')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  print("Modeを選択してください。　[通常RMS = 1][移動RMS = 0]")
  i = int(input())
  listener = Emg(mode=i) #emgクラスのインスタンス (mode0 = Moving_RMS) (mode1 = RMS)
  print("適当なキー入力で取得開始します")
  key = ord(getch())

  try:
    start = time.time()
    while hub.run(listener.on_event, 100):
      current = time.time()
      t = float(current - start)
      if listener.stop == 1:
        print("学習データ取得を終了します。　作業時間" ,t,"秒")
        break

  except KeyboardInterrupt:
    # Ctrl-C を捕まえた！
    print('interrupted!')
    # なにか特別な後片付けが必要ならここに書く
    sys.exit(0)

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()