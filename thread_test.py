import logging
import threading
import numpy as np
import time
from ctypes import windll


from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import serial
from PIL import Image

flag = True
def wait_input():
    global flag
    input()
    windll.winmm.timeEndPeriod(1)
    flag = False
    

def worker1(a, lock):
     #_____cnn_init________
    # データの保存先(自分の環境に応じて適宜変更)
    SAVE_DATA_DIR_PATH = "C:/Users/usui0/Desktop/2021_sekilab_data/demo/"
    # ラベル
    print("ok")
    labels =['oya', 'hitosashi', 'naka', 'kusuri', 'ko',"mu"]
    # self.labels =['oya', 'hitosashi', 'naka', 'kusuri', 'ko',"mu"]
    # 保存したモデル構造の読み込み
    model = model_from_json(open(SAVE_DATA_DIR_PATH+"lawmodel.json", 'r').read())
    print("ok")
    # 保存した学習済みの重みを読み込み
    model.load_weights(SAVE_DATA_DIR_PATH + "lawweight.h5")
    print("モデルと重みの読み込み完了")  

    while flag:
        pil_image = Image.fromarray(np.rot90(np.uint8(a)))
        img=img_to_array(pil_image)
        img = img.astype('float32')/255.0
        img = np.array([img])
        y_pred = model.predict(img)
        number_pred = np.argmax(y_pred) 
        # with lock:
        #     i=a[0][0]
        #     a[0][0]=i+1
        #     #print("th1",a)
        print("認識結果",labels[int(number_pred)])
        time.sleep(0.02)

def worker2(a, lock):
    gemg = np.ones((1,8))
    while flag:
        with lock:
            if a.shape[0] <= 101:
                a = np.append(a,gemg,axis=0)
            else:
                a = np.delete(a, 1, 0)

            # i = a[0][1]
            # a[0][1]=i+1       
            #print("th2",a)
        time.sleep(1)
        

def main():   
    windll.winmm.timeBeginPeriod(1) 
    a = np.zeros((100,8))
    lock = threading.Lock()
    th = threading.Thread(target=wait_input)

    t1 = threading.Thread(target=worker1, args=(a, lock))
    t2 = threading.Thread(target=worker2, args=(a, lock))
    th.start()
    t1.start()
    t2.start()
    t1.join()
    t2.join()

#main関数を最初に動かすおまじない
if __name__ == '__main__':
  main()