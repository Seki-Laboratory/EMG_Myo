#モジュールmyoをインポート
import myo
import time
import sys
import numpy as np
import csv
from msvcrt import getch
import winsound


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
    # print("ジェスチャ",self.label,"の学習データを取得します。")
  
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
          print("ジェスチャ",self.label,"の学習データを取得しました。  [続行 = Enter][終了 = Esc] ")
          # winsound.PlaySound("sound/3.wav", winsound.SND_FILENAME)
          while True:
            key = ord(getch())
            if key == 13:

              self.label += 1
              print("ジェスチャ",self.label,"の学習データを取得します。")
              event.device.stream_emg(True)
              self.i = 0
              break
            elif key == 27:
              print("学習データの取得終了します。")
              self.stop = 1
              break

        self.rms = np.zeros((1,8))
        self.j = 0

      self.j += 1

#main関数
def main():

  myo.init(bin_path=r'./bin')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  hub1 = myo.Hub()
  listener = Emg(mode=1) #emgクラスのインスタンス (mode0 = Moving_RMS) (mode1 = RMS)
  listener1 = myo.ApiDeviceListener()
  # winsound.PlaySound("sound/1.wav", winsound.SND_FILENAME)
  # winsound.PlaySound("sound/2.wav", winsound.SND_FILENAME)
  print("学習データ取得システム起動しました")
  with hub1.run_in_background(listener1.on_event):
    print("Waiting for a Myo to connect ...")
    # winsound.PlaySound("sound/check.wav", winsound.SND_FILENAME)
    device = listener1.wait_for_single_device(2)
  if not device:
    print("No Myo connected after 2 seconds.")
    # winsound.PlaySound("sound/error.wav", winsound.SND_FILENAME)
    return
  else:
    # winsound.PlaySound("sound/ok.wav", winsound.SND_FILENAME)
    print("適当なキー入力で取得を開始します")
    key = ord(getch())



  try:

    start = time.time()
    while hub.run(listener.on_event, 100) :
      current = time.time()
      t = float(current - start)
      if listener.stop == 1 or not device:
        print("作業時間" ,t,"秒")
        # winsound.PlaySound("sound/ed1.wav", winsound.SND_FILENAME)
        print("お疲れ様でした")
        break

  except KeyboardInterrupt:
    # Ctrl-C を捕まえた！
    print('interrupted!')
    # なにか特別な後片付けが必要ならここに書く
    sys.exit(0)

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()