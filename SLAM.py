#モジュールmyoをインポート
import myo
import time
import sys
import numpy as np
from msvcrt import getch
import winsound
import pygame
from pygame.locals import *

#Emgクラス　サンプリング周波数200でデータを取得するクラス
class Emg(myo.DeviceListener):

  def __init__(self,mode):
    print("class Emg instanced mode=",mode)
    self.add =  np.zeros((1,8)) 
    self.mode = mode  
    pygame.init()                                   # Pygameの初期化
    self.screen = pygame.display.set_mode((400, 400))    # 大きさ400*300の画面を生成
    pygame.display.set_caption("Test")    

  def on_connected(self, event):
      event.device.stream_emg(True)

  def on_emg(self,event):
    self.emg = np.array(event.emg)**2
    self.screen.fill((0,0,0))        # 画面を黒色(#000000)に塗りつぶし


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
      if sqrt[4] <= 27:
        pygame.draw.ellipse(self.screen,(0,100,0),(175,25,50,50),5)
      else:
        pygame.draw.ellipse(self.screen,(0,100,0),(175,25,50,50))
      if sqrt[5] <= 21:
        pygame.draw.ellipse(self.screen,(0,100,0),(281,69,50,50),5)
      else:
        pygame.draw.ellipse(self.screen,(0,100,0),(281,69,50,50))
      if sqrt[6] <= 39:
        pygame.draw.ellipse(self.screen,(0,100,0),(325,175,50,50),5)
      else:
        pygame.draw.ellipse(self.screen,(0,100,0),(325,175,50,50))

      if sqrt[7] <= 37:
        pygame.draw.ellipse(self.screen,(0,100,0),(281,281,50,50),5)
      else:
        pygame.draw.ellipse(self.screen,(0,100,0),(281,281,50,50)) 

      if sqrt[0] <= 30:
        pygame.draw.ellipse(self.screen,(0,100,0),(175,325,50,50),5)
      else:
        pygame.draw.ellipse(self.screen,(0,100,0),(175,325,50,50))

      if sqrt[1] <= 40:
        pygame.draw.ellipse(self.screen,(0,100,0),(69,281,50,50),5)
      else:
        pygame.draw.ellipse(self.screen,(0,100,0),(69,281,50,50))

      if sqrt[2] <= 27:
        pygame.draw.ellipse(self.screen,(0,100,0),(25,175,50,50),5)
      else:
        pygame.draw.ellipse(self.screen,(0,100,0),(25,175,50,50))
      if sqrt[3] <= 27:
        pygame.draw.ellipse(self.screen,(0,100,0),(69,69,50,50),5)
      else:
        pygame.draw.ellipse(self.screen,(0,100,0),(69,69,50,50))


      
      
      
      

      pygame.display.update()     # 画面を更新




#main関数
def main():

  myo.init(bin_path=r'./bin')
  hub = myo.Hub()  #myoモジュールのHubクラスのインスタンス
  listener = Emg(mode=0) #emgクラスのインスタンス (mode0 = Moving_RMS) (mode1 = RMS)


  try:

    start = time.time()
    while hub.run(listener.on_event, 100) :
      current = time.time()
      t = float(current - start)
      for event in pygame.event.get():
        if event.type == QUIT:  # 閉じるボタンが押されたら終了
            pygame.quit()       # Pygameの終了(画面閉じられる)
            sys.exit()
      # if t>=5 :
      #   print("作業時間" ,t,"秒")
      #   # winsound.PlaySound("sound/ed1.wav", winsound.SND_FILENAME)
      #   print("お疲れ様でした")
      #   break

  except KeyboardInterrupt:
    # Ctrl-C を捕まえた！
    print('interrupted!')
    # なにか特別な後片付けが必要ならここに書く
    pygame.quit()       # Pygameの終了(画面閉じられる)
    sys.exit(0)

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()