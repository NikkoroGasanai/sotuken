#入力された音をそのまま流し、波形とスペクトルとカラレーション周波数のスペクトルを出す
#たぶんラグはpyaudioのせい

import copy
import pyaudio
import numpy as np
from scipy import signal
from scipy.signal import argrelmax
import matplotlib.pyplot as plt
import struct
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore



class AudioFilter():
    CHANNELS = 1
    RATE = 16000
    CHUNK = 2048
    START = 0
    N = CHUNK
    FORMAT = pyaudio.paFloat32
    WAVE_RANGE = 1
    SPECTRUM_RANGE = 50
    UPDATE_SECOND = 1

    NYQ = RATE / 2
    FILLEN = 10
    NUMTAPS = int(RATE*N+1)
    WAVETIME = range(START,START+N)
    FREQLIST = np.fft.fftfreq(N,d=1.0/RATE) 
    amplitudeSpectrumc=[0]*N
    colflist=[]
    COLFCNT=10#求めるカラレーション周波数の数

    def __init__(self):
        # オーディオ設定
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
                        format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        output=True,
                        input=True,
                        stream_callback=self.callback,
                        frames_per_buffer = self.CHUNK)
        
        # pyqtgraph
        self.app = QtGui.QApplication([])
        self.app.quitOnLastWindowClosed()
        self.win = QtGui.QMainWindow()
        self.win.setWindowTitle("SpectrumAnalyzer")
        self.win.resize(1600, 600)
        self.centralwid = QtGui.QWidget()
        self.win.setCentralWidget(self.centralwid) 
        self.lay = QtGui.QVBoxLayout()
        self.centralwid.setLayout(self.lay)
        # グラフ１
        self.plotwid1 = pg.PlotWidget(name="wave")
        self.plotitem1 = self.plotwid1.getPlotItem()
        self.plotitem1.setMouseEnabled(x = False, y = False) 
        #self.plotitem1.setYRange(0, self.SPECTRUM_RANGE)#スペクトル用
        #self.plotitem1.setXRange(0, self.RATE / 2, padding = 0)
        self.plotitem1.setYRange(self.WAVE_RANGE * -1, self.WAVE_RANGE * 1)#波形用
        self.plotitem1.setXRange(self.START, self.START + self.N, padding = 0)
        # グラフ2
        self.plotwid2 = pg.PlotWidget(name="spectrum")
        self.plotitem2 = self.plotwid2.getPlotItem()
        self.plotitem2.setMouseEnabled(x = False, y = False) 
        self.plotitem2.setYRange(0, self.SPECTRUM_RANGE)
        self.plotitem2.setXRange(0, self.RATE / 1.8, padding = 0)
        # グラフ３
        self.plotwid3 = pg.PlotWidget(name="Coloration spectrum")
        self.plotitem3 = self.plotwid3.getPlotItem()
        self.plotitem3.setMouseEnabled(x = False, y = False) 
        self.plotitem3.setYRange(0, self.SPECTRUM_RANGE)
        self.plotitem3.setXRange(0, self.RATE / 1.8, padding = 0)
        # Axis
        self.specAxis1 = self.plotitem1.getAxis("bottom")
        self.specAxis1.setLabel("Time [sample]")
        self.specAxis2 = self.plotitem2.getAxis("bottom")
        self.specAxis2.setLabel("Frequency [Hz]")
        self.specAxis3 = self.plotitem3.getAxis("bottom")
        self.specAxis3.setLabel("Frequency [Hz]")

        self.curve_wave = self.plotitem1.plot()
        self.curve_spectrum = self.plotitem2.plot()        
        self.curve_coloration = self.plotitem3.plot()
        self.lay.addWidget(self.plotwid1)
        self.lay.addWidget(self.plotwid2)
        self.lay.addWidget(self.plotwid3)

        self.win.show()
        #アップデート
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.UPDATE_SECOND)

    
        

    def update(self):
        data = np.frombuffer(self.outdata, np.float32)
        wave_figure = data[self.START:self.START + self.N]

        tria=[n/self.N for n in list(range(0,self.N))]#三角窓関数
        cha= np.fft.fft([x*y for x,y in zip(tria,data[self.START:self.START + self.N])])#CHA
        x  = np.fft.fft(data[self.START:self.START + self.N])#普通のフーリエ
        
        self.amplitudeSpectrumc= [np.sqrt(c.real ** 2 + c.imag ** 2) for c in cha]#CHA
        amplitudeSpectrum = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in x]#フーリエ
        self.curve_wave.setData(self.WAVETIME, wave_figure)
        self.curve_spectrum.setData(self.FREQLIST, amplitudeSpectrum)
        #self.curve_coloration.setData(self.FREQLIST, colf)
        
        
    # コールバック関数（再生が必要なときに呼び出される）
    def callback(self, in_data, frame_count, time_info, status):
        out_data = in_data

        #カラレーション周波数の計算
        aScf=argrelmax(np.array(self.amplitudeSpectrumc[100:round(self.N/2)]))[0]#ピークのindex（周波数的な）（list
        aScp=[self.amplitudeSpectrumc[i] for i in aScf]#ピークの振幅(list)
        colf=[]#カラレーション周波数
        for i in range(self.COLFCNT):
            colf.append(100+self.amplitudeSpectrumc.index(max(aScp,default=0)))
            try:
                aScp.remove(max(aScp,default=0))
            except:
                pass
        self.colflist=self.colflist+colf
        if len(self.colflist)>self.COLFCNT*10:
            self.colflist=self.colflist[-1*self.COLFCNT:]
        pntcolf=[round(i * self.RATE / self.N) for i in colf]
        print(('{: 12}, '*len(pntcolf)).format(*pntcolf))
        

            
        #10の周波数でのノッチフィルタ(重すぎて無理)
        '''if max(amplitudeSpectrumc)>50:
            firfreq=sum([[(i+self.FILLEN)/self.RATE,i/self.RATE,(i-self.FILLEN)/self.RATE] for i in colf],[])
            firfreq.insert(0,0)
            firfreq.append(1)
            firfreq.sort()
            a=[1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1]
            if len(a)==len(firfreq):
                fir=signal.firwin2(self.NUMTAPS,firfreq,a)
        '''
        
        self.outdata=in_data
        return (out_data, pyaudio.paContinue)

    def close(self):
        self.p.terminate()

if __name__ == "__main__":
    af = AudioFilter()
    af.stream.start_stream()

    QtGui.QApplication.instance().exec_()
    #以下動作しない？
    

    ## ノンブロッキングなので好きなことをしていていい場所
    #while af.stream.is_active():
    #    time.sleep(0.1)
    #    print()

    # ストリーミングを止める場所
    af.stream.stop_stream()
    af.stream.close()
    af.close()
