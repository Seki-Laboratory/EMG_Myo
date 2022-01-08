#モジュールmyoをインポート
import myo
import time
import sys
import numpy as np
import csv
from msvcrt import getch
import cv2
import matplotlib.pyplot as plt
import winsound
import os
import re



#Emgクラス　サンプリング周波数200でデータを取得するクラス
class Emg(myo.DeviceListener):


  def __init__(self,mode,n):
    print("class Emg instanced mode=",mode)
    self.rms = np.zeros((1,8))   
    self.add =  np.zeros((1,8)) 
    self.mode = mode
    self.n = n
    self.i = 0
    self.j = 0
    self.label = int(0)
    self.stop = 0
    self.geta128 = [128,128,128,128,128,128,128,128]
    # print("ジェスチャ",self.label,"の学習データを取得します。")

  def list_csv(directory, ext='csv'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]
  
  def on_connected(self, event):
      event.device.stream_emg(True)

  def on_emg(self,event):
    SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/demo/"
    JES = "hitosashi"

    self.emg = np.array(event.emg)
    #myo_armbandは8bit(-128~127)の値を返すので127足して基準位置をずらします。
    gemg = self.geta128+self.emg
    print(gemg)
    if self.i <= self.n-1:
        with open(SAVE_DATA_DIR_PATH+JES+"/"+JES+'_emgdata'+str(self.label+202)+'.csv', 'a') as f:
            writer = csv.writer(f, lineterminator='\n') # 行末は改行
            law_emg = np.append(gemg,self.label)
            writer.writerow(law_emg)
        self.i += 1
    elif self.i >= self.n:
        event.device.stream_emg(False)
        print("ジェスチャ",self.label,"の学習データを取得しました。  [続行 = Enter][終了 = Esc] ")

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


#main関数
def main():

  myo.init(bin_path=r'./bin')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  hub1 = myo.Hub()
  listener = Emg(mode=0,n=100) #emgクラスのインスタンス (mode0 = Moving_RMS) (mode1 = RMS)
  listener1 = myo.ApiDeviceListener()

  print("学習データ取得システム起動しました")
  with hub1.run_in_background(listener1.on_event):
    print("Waiting for a Myo to connect ...")

    device = listener1.wait_for_single_device(2)
  if not device:
    print("No Myo connected after 2 seconds.")

    return
  else:

    print("適当なキー入力で取得を開始します")
    key = ord(getch())



  try:

    start = time.time()
    while hub.run(listener.on_event, 5) :
      current = time.time()
      t = float(current - start)
      # print(t)
      if listener.stop == 1 or not device:
        print("作業時間" ,t,"秒")

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