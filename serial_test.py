import serial
import time


#シリアル通信(PC⇔Arduino)
ser = serial.Serial()
ser.port = "COM3"     #デバイスマネージャでArduinoのポート確認
ser.baudrate = 115200 #Arduinoと合わせる
ser.setDTR(False)     #DTRを常にLOWにしReset阻止
ser.open()            #COMポートを開く
ser.write(b'1')       #送りたい内容をバイト列で送信
print("send")
ser.close()           #COMポートを閉じる
time.sleep(2)
start = time.time()
ser.open()  
ser.write(b'0')       #送りたい内容をバイト列で送信
ser.close() 
current = time.time()
print(float(current - start))
print("send")
time.sleep(2)
with serial.Serial('COM3',115200, timeout=1) as ser:
    ser.write(b'1')       #送りたい内容をバイト列で送信
print("send")
time.sleep(2)
with serial.Serial('COM3',115200, timeout=1) as ser:
    ser.write(b'0')       #送りたい内容をバイト列で送信
print("send")