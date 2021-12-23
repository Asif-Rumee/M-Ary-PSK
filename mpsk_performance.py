"""
Performance analysis of M-ary PSK over AWGN channel by SER vs Es/N0 chart.
Author: Asif Rahman Rumee
Date: 23/12/2021

"""
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import erfc
from mpsk_modem import mpsk_modem, awgn

nSym = 10**4
EbN0dBs = np.arange(start=-2,stop = 26, step = 2)
arrayOfM = [2,4,8,16,32]
colors = ['k', 'b', 'r', 'm', 'y']
fig, ax = plt.subplots(nrows=1,ncols=1)

for i, M in enumerate(arrayOfM):
    k=np.log2(M)
    EsN0dBs = 10*np.log10(k)+EbN0dBs 
    SER = np.zeros(len(EbN0dBs)) 
    inputSyms = np.random.randint(low=0, high = M, size=nSym)
    modem = mpsk_modem(M) 
    modulatedSyms = modem.modulate(inputSyms)

    for j,EsN0dB in enumerate(EsN0dBs):
        receivedSyms = awgn(modulatedSyms,EsN0dB)
        detectedSyms = modem.demodulate(receivedSyms)
        SER[j] = np.sum(detectedSyms != inputSyms)/nSym

    ax.semilogy(EbN0dBs,SER,color=colors[i],marker='o',linestyle='-',label='M = '+str(M))
    
ax.set_xlabel('Es/N0(dB)')
ax.set_ylabel('SER ($P_s$)')
ax.set_title('Performance analysis of Symbol Error Rate for MPSK over AWGN')
ax.legend()
plt.show()