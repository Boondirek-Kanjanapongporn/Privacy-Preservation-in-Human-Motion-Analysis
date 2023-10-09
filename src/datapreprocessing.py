import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram, windows
from math import floor
from helperfunctions import *

# Data reading part and load data
filename = 'src/data/1P03A01R1.dat'
fd = open(filename,'r')  
lines = fd.readlines()
radarData = np.array([complex_converter(line.strip()) for line in lines])
fd.close()
fc = radarData[0]
Tsweep = radarData[1] / 1000  # in seconds
NTS = int(radarData[2])
Bw = radarData[3]
Data = radarData[4:]
fs = NTS / Tsweep
record_length = len(Data) / NTS * Tsweep
nc = record_length / Tsweep

# Reshape data into chirps and plot Range-Time
Data_time = np.reshape(Data, (NTS, int(nc)), order='F')
win = np.ones_like(Data_time)

# Part taken from Ancortek code for FFT and IIR filtering
tmp = np.fft.fftshift(np.fft.fft(Data_time * win, axis=0), axes=0)
Data_range = tmp[int(NTS/2):, :]
ns = oddnumber(np.shape(Data_range)[1]) - 1
Data_range_MTI = np.zeros_like(Data_range, dtype=complex)
b, a = butter(4, 0.0075, 'high')
for k in range(Data_range.shape[0]):
    Data_range_MTI[k, :ns] = lfilter(b, a, Data_range[k, :ns])
freq = np.arange(ns) * fs / (2 * ns)
range_axis = (freq * 3e8 * Tsweep) / (2 * Bw)
Data_range_MTI = Data_range_MTI[1:, :]
Data_range = Data_range[1:, :]
# plt.figure()
# img = plt.imshow(20 * np.log10(np.abs(Data_range_MTI)), aspect='auto', cmap='jet', origin='lower')
# plt.xlabel('No. of Sweeps')
# plt.ylabel('Range bins')
# plt.title('Range Profiles after MTI filter')
# plt.colorbar(label='Amplitude (dB)')
# plt.ylim([1, 100])
# clim = img.get_clim()
# plt.clim(clim[1]-60, clim[1])
# plt.show()

# Spectrogram processing for 2nd FFT to get Doppler
bin_indl = 10
bin_indu = 30
MD = {
    "PRF": 1/Tsweep,
    "TimeWindowLength": 200,
    "OverlapFactor": 0.95,
    "Pad_Factor": 4,
}
MD["OverlapLength"] = round(MD["TimeWindowLength"] * MD["OverlapFactor"])
MD["FFTPoints"] = MD["Pad_Factor"] * MD["TimeWindowLength"]
MD["DopplerBin"] = MD["PRF"] / MD["FFTPoints"]
MD["DopplerAxis"] = np.linspace(-MD["PRF"]/2, MD["PRF"]/2, MD["FFTPoints"], endpoint=False)
MD["WholeDuration"] = Data_range_MTI.shape[1] / MD["PRF"]
MD["NumSegments"] = int((Data_range_MTI.shape[1] - MD["TimeWindowLength"]) / floor((MD["TimeWindowLength"] * (1 - MD["OverlapFactor"]))))
Data_spec_MTI2 = 0
Data_spec2 = 0
win = windows.hamming(MD["TimeWindowLength"])
scaling_factor = 2/(fs * np.sum(win**2))
for RBin in range(bin_indl, bin_indu + 1):
    f, t, Sxx = spectrogram(Data_range_MTI[RBin - 1, :], nperseg=MD["TimeWindowLength"], noverlap=MD["OverlapLength"], nfft=MD["FFTPoints"], return_onesided=False)
    Data_spec_MTI2 += np.abs(np.fft.fftshift(Sxx, axes=0))
    f, t, Sxx = spectrogram(Data_range[RBin - 1, :], nperseg=MD["TimeWindowLength"], noverlap=MD["OverlapLength"], nfft=MD["FFTPoints"], return_onesided=False)
    Data_spec2 += np.abs(np.fft.fftshift(Sxx, axes=0))

MD["TimeAxis"] = np.linspace(0, MD["WholeDuration"], Data_spec_MTI2.shape[1])
Data_spec_MTI2=np.flipud(Data_spec_MTI2)

# Scaling to range 104- 140
Data_spec_MTI2_processed = 20 * np.log10(np.abs(Data_spec_MTI2))
maxV = np.max(Data_spec_MTI2_processed)  
minV = np.min(Data_spec_MTI2_processed)
Data_spec_MTI2_scaled = (Data_spec_MTI2_processed - minV) / (maxV - minV) * (140 - 100) + 100

plt.figure()
plt.imshow(Data_spec_MTI2_scaled, aspect='auto', cmap='jet', extent=[MD["TimeAxis"][0], MD["TimeAxis"][-1], -MD["DopplerAxis"][0]*3e8/2/5.8e9, -MD["DopplerAxis"][-1]*3e8/2/5.8e9])
plt.colorbar()
plt.ylim(-6, 6)
plt.xlabel('Time[s]')
plt.ylabel('Velocity [m/s]')
plt.title(filename.split('/')[-1])
plt.show()
