"""
Design of modulator and demodulator of M-ary PSK over AWGN channel.
Author: Asif Rahman Rumee
Date: 23/12/2021

"""
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

class mpsk_modem:
    def __init__(self, constellation_points):
        if (constellation_points<2) or ((constellation_points & (constellation_points -1))!=0):
            raise ValueError('Constellation point should be a power of 2')
        
        symbols = np.arange(0,constellation_points)
        I = 1/np.sqrt(2)*np.cos(symbols/constellation_points*2*np.pi)
        Q = 1/np.sqrt(2)*np.sin(symbols/constellation_points*2*np.pi)
        self.constellation = I + 1j*Q #reference constellation
        self.constellation_points = constellation_points

    def modulate(self,input_symbols):
        if isinstance(input_symbols,list):
            input_symbols = np.array(input_symbols)
        if not (0 <= input_symbols.all() <= self.constellation_points-1):
            raise ValueError('Values for inputSymbols are outside the range 0 to constellation_points-1')
        modulated_vector = self.constellation[input_symbols]
        return modulated_vector

    def demodulate(self,received_symbols):
        if isinstance(received_symbols,list):
            received_symbols = np.array(received_symbols)

        detected_symbols= self.min_distance_detector(received_symbols)
        return detected_symbols

    def min_distance_detector(self,received_symbols):
        XA = np.column_stack((np.real(received_symbols),np.imag(received_symbols)))
        XB=np.column_stack((np.real(self.constellation),np.imag(self.constellation)))
        euclid_distances = cdist(XA,XB,metric='euclidean') 
        detected_symbols = np.argmin(euclid_distances,axis=1)
        return detected_symbols

def awgn(s,SNRdB,L=1):
    """
    AWGN channel
    """
    gamma = 10**(SNRdB/10)
    if s.ndim==1:
        P=L*sum(abs(s)**2)/len(s)
    else:
        P=L*sum(sum(abs(s)**2))/len(s)
    N0=P/gamma
    if isrealobj(s):
        n = sqrt(N0/2)*standard_normal(s.shape)
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal
    return r