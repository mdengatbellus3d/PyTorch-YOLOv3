import time
import threading
from multiprocessing import Queue, Process
import tkinter as tk

import smtplib

from pytorchyolo import detect

# the detect arguments
IMAGE_FOLDER_PATH = 'C:\\max\\truck\\testimages\\'
WEIGHTS_FILE_PATH = 'C:\\max\\truck\\training\\yolo3-256-16\\yolov3_ckpt_660.pth'
CONF_THRES = '0.1'


# to compose the detect.py command line arguments...
detectpyCommandline = [
    # 'yolo-detect',
    '--images', IMAGE_FOLDER_PATH, '--model', 'ttruck/config/yolov3-custom.cfg', '--classes',
    'ttruck/config/classes.names', '--weights', WEIGHTS_FILE_PATH, '--conf_thres', CONF_THRES]

# the main window


class MainWindow:
    def __init__(self):
        self.m = tk.Tk()
        self.m.title('Tesla Truck Detector')
        self.m.geometry('640x640')

        # lib
        self.gmail_server = None

        # app states
        self.data = None
        self.canvas = None
        self.detectionThreadCreated = False
        self.nextQueryCount = 0

        # tkinter variable bindings
        self.notificationUpperPrice = tk.DoubleVar()
        self.notificationLowerPrice = tk.DoubleVar()
        self.notificationEmail = tk.StringVar()
        self.appState = tk.StringVar()
        self.lastestPrice = tk.StringVar()

        # watch BTC by default
        # self.tickerSymbol.set("BTC")

        #
        # create UI elements
        #
        leftSideWidth = 25

        frame2 = tk.Frame(self.m)
        self.startButton = tk.Button(
            frame2, text='Start Monitor', command=self.startMonitor)
        self.stopButton = tk.Button(
            frame2, text='Stop Monitor', command=self.stopMonitor)
        self.startButton.pack(side=tk.LEFT, padx=5)
        self.stopButton.pack(side=tk.LEFT, padx=5)
        self.stopButton.config(state='disabled')
        frame2.pack(pady=5)

        frame3 = tk.Frame(self.m)
        tk.Label(frame3, text='Notification Upper Price:',
                 width=leftSideWidth).pack(side=tk.LEFT)
        notificationPrice = tk.Entry(
            frame3, textvariable=self.notificationUpperPrice).pack(side=tk.LEFT)
        frame3.pack(pady=5)

        frame4 = tk.Frame(self.m)
        tk.Label(frame4, text='Notification Lower Price:',
                 width=leftSideWidth).pack(side=tk.LEFT)
        tk.Entry(frame4, textvariable=self.notificationLowerPrice).pack(
            side=tk.LEFT)
        frame4.pack(pady=5)

        frame5 = tk.Frame(self.m)
        tk.Label(frame5, text='Notification Email:',
                 width=leftSideWidth).pack(side=tk.LEFT)
        tk.Entry(frame5, textvariable=self.notificationEmail).pack(side=tk.LEFT)
        frame5.pack(pady=5)

        frame6 = tk.Frame(self.m)
        tk.Button(frame6, text='Quit',
                  command=self.m.destroy).pack(side=tk.LEFT)
        frame6.pack(pady=5)

        frame7 = tk.Frame(self.m)
        tk.Label(frame7, textvariable=self.appState,
                 fg='green').pack(side=tk.LEFT)
        frame7.pack(pady=5)

        frame8 = tk.Frame(self.m)
        tk.Label(frame8, textvariable=self.lastestPrice,
                 fg='green').pack(side=tk.LEFT)
        frame8.pack(pady=5)

        self.initApiSessions()

        self.startUiRefreshTimer()

        tk.mainloop()

    def initApiSessions(self):
        try:
            pass
        except Exception as e:
            pass

    def startUiRefreshTimer(self):
        def timer():
            # plot latest graph
            if self.nextQueryCount == 10:
                self.plot()

            # update app state texts
            if self.monitoring:
                if self.nextQueryCount > 0:
                    self.nextQueryCount -= 1
                    self.appState.set(
                        'Next query in {} seconds...'.format(self.nextQueryCount))
            else:
                self.appState.set('Monitor has stopped.')

            # trigger the timer again
            self.m.after(1000, timer)

        # run first time
        timer()

    def startMonitor(self):
        self.startButton.config(state='disabled')
        self.stopButton.config(state='active')

        detect.setRunningState(False)

        if not self.detectionThreadCreated:
            threading.Thread(target=self.detectThreadFunction).start()
            self.detectionThreadCreated = True

    def stopMonitor(self):
        if not self.monitoring:
            return
        self.monitoring = False
        self.startButton.config(state='active')
        self.stopButton.config(state='disabled')

        detect.setRunningState(True)

    def detectThreadFunction(self):
        detect.run(detectpyCommandline)
        # print('detection thread started.')

    def plot(self):
        pass

    def sendNotification(self, email, price, notification_point):
        pass


def run():
    window = MainWindow()


if __name__ == '__main__':
    run()
