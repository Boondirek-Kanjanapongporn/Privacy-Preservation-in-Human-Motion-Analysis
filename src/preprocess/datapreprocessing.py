import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram
from math import floor
from preprocess.preprocesshelperfunctions import *
import tensorflow as tf
from sklearn import preprocessing

def preprocess(folder, filename, showFig1, showFig2, store, normalize):
    # Data reading part and load data
    # filepath = getfilepath1(filename)
    filepath = getfilepath2(folder, filename)
    fd = open(filepath,'r')
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

    # Plot raw data
    # plt.figure(figsize=(8,6))
    # plt.plot(np.abs(Data))
    # # plt.title('Raw Radar Data')
    # plt.xlabel('Sample Index', fontsize=22, fontweight='bold')
    # plt.ylabel('Magnitude', fontsize=22, fontweight='bold')
    # plt.xticks(fontsize=14, fontweight='bold')
    # plt.yticks(fontsize=14, fontweight='bold')
    # plt.show(block=True)

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

    # Plot figure 1
    if showFig1:
        plt.figure()
        img = plt.imshow(20 * np.log10(np.abs(Data_range_MTI)), aspect='auto', cmap='jet', origin='lower')
        plt.xlabel('No. of Sweeps', fontsize=18, fontweight='bold')
        plt.ylabel('Range bins', fontsize=18, fontweight='bold')
        plt.title('Range Profiles after MTI filter')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(label='Amplitude (dB)', size=18, weight='bold')
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.ylim([1, 62])
        clim = img.get_clim()
        plt.clim(clim[1]-60, clim[1])
        plt.show(block=False) if showFig2 else plt.show()

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
    for RBin in range(bin_indl, bin_indu + 1):
        f, t, Sxx = spectrogram(Data_range_MTI[RBin - 1, :], nperseg=MD["TimeWindowLength"], noverlap=MD["OverlapLength"], nfft=MD["FFTPoints"], return_onesided=False)
        Data_spec_MTI2 += np.abs(np.fft.fftshift(Sxx, axes=0))
        f, t, Sxx = spectrogram(Data_range[RBin - 1, :], nperseg=MD["TimeWindowLength"], noverlap=MD["OverlapLength"], nfft=MD["FFTPoints"], return_onesided=False)
        Data_spec2 += np.abs(np.fft.fftshift(Sxx, axes=0))
    MD["TimeAxis"] = np.linspace(0, MD["WholeDuration"], Data_spec_MTI2.shape[1])
    Data_spec_MTI2=np.flipud(Data_spec_MTI2)
    Data_spec_MTI2 = np.float32(Data_spec_MTI2)

    # Process Data_spec_MTI2
    Data_spec_MTI2_processed = 20 * np.log10(np.abs(Data_spec_MTI2))

    # For Normalization
    if normalize:
        data_min = np.min(Data_spec_MTI2_processed)
        data_max = np.max(Data_spec_MTI2_processed)
        Data_spec_MTI2_processed = (Data_spec_MTI2_processed - data_min) / (data_max - data_min)
        # --------------------------
        # Data_spec_MTI2_processed = Data_spec_MTI2_processed/np.linalg.norm(Data_spec_MTI2_processed)
        # --------------------------
        # min_max_scaler = preprocessing.MinMaxScaler()
        # Data_spec_MTI2_processed = min_max_scaler.fit_transform(Data_spec_MTI2_processed) 

    rows, cols = Data_spec_MTI2_processed.shape
    Data_spec_MTI2_processed = Data_spec_MTI2_processed.reshape(rows, cols, 1)
    
    # --------------------------
    # For File Storing (800, 481)
    # if store and cols > 481:
    #     Data_spec_MTI2_processed = tf.image.resize_with_pad(Data_spec_MTI2_processed, target_height=800, target_width=481)
    # --------------------------
    # For File Storing (400, 240)
    if store and cols > 240:
        Data_spec_MTI2_processed = tf.image.resize_with_pad(Data_spec_MTI2_processed, target_height=400, target_width=240)
    
    # Plot figure 2
    if showFig2:
        plt.figure()
        img = plt.imshow(Data_spec_MTI2_processed, aspect='auto', cmap='jet', extent=[MD["TimeAxis"][0], MD["TimeAxis"][-1], -MD["DopplerAxis"][0]*3e8/2/5.8e9, -MD["DopplerAxis"][-1]*3e8/2/5.8e9])
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(label='Amplitude (dB)', size=18, weight='bold')
        plt.ylim(-6, 6)
        plt.xlabel('Time[s]', fontsize=18, fontweight='bold')
        plt.ylabel('Velocity [m/s]', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.tight_layout()
        clim = img.get_clim()
        if normalize:
            plt.clim(clim[1]-0.6, clim[1])
        else:
            plt.clim(clim[1]-80, clim[1])
        # plt.title(filepath.split('/')[-1])
        plt.show()

    return Data_spec_MTI2_processed
