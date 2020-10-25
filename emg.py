from __future__ import print_function
import collections
import myo
import time
import sys


class Emg(myo.DeviceListener):

  def on_arm_synced(self, event):
    event.device.stream_emg(False)

  def on_emg(self, event):
    emg = event.emg




def main():
  myo.init(sdk_path=r'C:\work\myo-sdk-win-0.9.0-main')
  hub = myo.Hub()
  listener = Emg()
  while hub.run(listener.on_event, 500):
    pass

if __name__ == '__main__':
  main()