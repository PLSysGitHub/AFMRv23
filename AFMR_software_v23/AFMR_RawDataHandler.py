# -*- coding: utf-8 -*-
"""
Updated script created on Mon Jul  3 15:56:06 2023
Last edit: August 24, 2023

@author: Xam, Giulia, Kiki, Hannes
"""

# Import necessary modules and libraries
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import csd
from scipy.fftpack import fft, fftfreq

from AFMR_FileHandler import RequestFolderPath, SortData, UserInput, FilenameBeadResults, ErrorFiles

#____________________________________________________________________________________________________________________________

CONVERSION_FACTOR_FORCE = 0.55 # Used to convert the measured intensity to force based on the system calibration

# Set to True to plot figures of raw data and fourier transform of frequency
TROUBLE_SHOOTING = True
PLOT_RAW_DATA = True
#____________________________________________________________________________________________________________________________

def main():
    """
    Main function that coordinates data processing and analysis
    """   
    
    # Initialize GUI which requests the folder path from the user
    request_folder = RequestFolderPath()
    user_folder = request_folder.folderpath()
    
    # Create folder for storing result files, figures and error files
    results_foldername = "Result per beadnumber"
    results_folder = request_folder.createresultsfolder(results_foldername)
    
    figures_foldername = "Figures raw data checks"
    figures_folder = request_folder.createfiguresfolder(figures_foldername)
    
    error_foldername = "No match frequency"
    error_folder = request_folder.createfiguresfolder(error_foldername)
    
    # Retrieve data from the tdms files within the chosen folder
    data = SortData(user_folder).tdms_files_dataframes()
    
    # Perform first step of AFMR analysis on the raw data
    analysis_raw_data = AFMR(data, user_folder, results_folder, figures_folder, error_folder)
    analysis_raw_data.rheology_results()
    
#____________________________________________________________________________________________________________________________

class PlotData:
    
    """ 
    Plot and save the raw data and Lissajous figures based on input frequency
    """
    
    def __init__(self, figures_folder, time, force, displacement, filename, groupname, input_freq):
        
        """
        Initialize the PlotData class.

        Args:
            figures_folder (str): Path to the folder where figures will be saved.
            time (array-like): Time data for the plot.
            force (array-like): Force data for the plot.
            displacement (array-like): Displacement data for the plot.
            filename (str): Original filename associated with the data.
            groupname (str): Name of the data group.
            input_freq (float): Input frequency associated with the data.
        """
        
        self.save_figures = figures_folder
        self.time = time
        self.displ = displacement
        self.force = force
        self.filename = filename
        self.groupname = groupname
        self.freq = input_freq
    
    def plot_raw_data(self):     
        
        """
        Plot and save raw data figure.
        """
            
        fig, ax = plt.subplots(figsize = (10, 6))
        ax1 = ax.twinx()
        
        force_plot = ax.plot(self.time, self.force, color = "darkmagenta")            
        displ_plot = ax1.plot(self.time, self.displ, color = "darkgreen")
        
        ax.set_xlabel("Time (s)", fontsize = 16)
        ax.set_ylabel("Force (pN)", fontsize = 16)
        ax.yaxis.label.set_color(force_plot[0].get_color())
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(0)
        
        ax1.set_ylabel("Displacement (nm)", fontsize = 16)
        ax1.yaxis.label.set_color(displ_plot[0].get_color())
        ax1.tick_params(axis='y', which='major', labelsize=14)
        ax1.set_ylim(np.min(self.displ))
        
        fig.suptitle(f"f = {self.freq} Hz", fontsize = 18)            
        figname = f"Raw data {self.filename}_{self.groupname}.png"
        fig.savefig(os.path.join(self.save_figures, figname), bbox_inches = "tight")
        
        fig.show()
            
    def plot_lissajous(self):
        
        """
        Plot and save Lissajous figure.
        """
        
        fig, ax = plt.subplots(figsize = (10, 6))
        
        ax.scatter(self.displ, self.force, color = "gray")
        ax.set_xlabel("Displacement (nm)", fontsize = 16)
        ax.set_ylabel("Force (pN)", fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        fig.suptitle(f"f = {self.freq} Hz", fontsize = 18)
        figname = f"Lissajous {self.filename}_{self.groupname}.png"
        fig.savefig(os.path.join(self.save_figures, figname), bbox_inches = "tight")
        
        fig.show()
        

#____________________________________________________________________________________________________________________________    

class ConvertIntensity:
    
    """
    Convert intensity data to force using provided parameters.
    """
    
    def __init__(self, voltage, mod):
        
        """
        Initialize the ConvertIntensity class.

        Args:
            voltage (float): Input voltage value.
            mod (float): Modulation value.
        """
        
        self.input_volt = voltage
        self.input_mod = mod
        
    def force_conversion(self, intensity):
        
        """
        Convert intensity data to force using provided parameters.

        Args:
            intensity (array-like): Intensity data to be converted.

        Returns:
            list: List of force values corresponding to the input intensity.
        """
        
        baseline_intensity = np.min(intensity)
        max_int = np.amax(intensity)
        baseline_volt = (1-self.input_mod/100)*self.input_volt
        
        conv_volt = (max_int - baseline_intensity) / (self.input_volt - baseline_volt)
        force = [CONVERSION_FACTOR_FORCE * (baseline_volt + (1 / (conv_volt)) * (i - baseline_intensity))**2 for i in intensity] 
        
        return force

#____________________________________________________________________________________________________________________________
        
class FourierTransform:
    
    """
    Apply a Fourier Transform on a signal to find the amplitude at a target frequency.
    """
    
    def __init__(self, time):
        
        """
        Initialize the FourierTransform class.

        Args:
            time (array-like): Time values associated with the signal.
        """
        
        self.time = time
                
    def signal_amplitude_fft(self, signal, target_freq):  
        
        """ 
        Apply fast Fourier transform to find the amplitude of a signal at a specific frequency.

        Args:
            signal (array-like): The input signal to analyze.
            target_freq (float): The target frequency at which to find the amplitude.

        Returns:
            tuple: A tuple containing the found frequency and its associated amplitude.
        """
        
        # Calculate the sampling frequency
        fs = np.shape(self.time[~np.isnan(self.time)])[0] / (np.max(self.time) - np.min(self.time))  
        samples = len(signal)
        
        # Apply Hanning window and remove mean from the signal
        signal = np.hanning(samples) * (signal - np.mean(signal)) 
        
        # Perform FFT with zero-padding
        fft_signal = fft(signal, n = 5 * samples) # Zero padded with 5 times the signal length
        frequencies = fftfreq(5 * samples) * fs
        amplitudes = 2 * (2 / samples * np.abs(fft_signal)) # Added factor 2 because of windowing
        
        for i in range(len(frequencies)):
            if frequencies[i] == 0:
                amplitudes[i] = 0
        
        # Search for the frequency and amplitude that most closely matches the input target_freq
        idx = np.argmin(np.abs(frequencies - target_freq))
        found_frequency = frequencies[idx]
        found_amplitude = amplitudes[idx]
        
        # Search for the index with the maximum amplitude for trouble shooting plot
        idx2 = np.argmax(amplitudes) 
                        
        if TROUBLE_SHOOTING:
            
            # Plot for trouble shooting
            fig, ax = plt.subplots(figsize = (10, 6))
            
            ax.plot(frequencies, amplitudes, linestyle = "dashdot", label = "FFT signal")
            ax.plot(found_frequency, found_amplitude, marker = "*", label = "Found frequency & amplitude")
            ax.plot(frequencies[idx2], amplitudes[idx2], marker = "o", label = "Max frequency & amplitude")
            ax.set_xlabel("Frequency (Hz)", fontsize = 16)
            ax.set_ylabel("Amplitude (a.u.)", fontsize = 16)
            ax.set_xlim([-2 * target_freq, 2 * target_freq])
            
            fig.suptitle("Target frequency = {0:} Hz \n FFT max frequency = {1:.3f} Hz & amplitude = {2:.3f}".format(target_freq, frequencies[idx2], amplitudes[idx2]), fontsize = 18)
            fig.legend(bbox_to_anchor = (1, 1), loc = "upper left")
            fig.tight_layout()
            fig.show()
            
        if target_freq < 0.09:            
            found_frequency = round(found_frequency, 2) # Round fft frequency for check
        else:
            found_frequency = round(found_frequency, 1) # Round fft frequency for check
        
        return found_frequency, found_amplitude # use this to determine the prefactor
        
 
#____________________________________________________________________________________________________________________________
       
class SignalAnalysis:
    
    """
    Perform spectral density calculations on signal and find frequency and amplitudes from the fourier transform
    """
    
    def __init__(self, time, force, displacement, input_freq):
        
        """
        Initialize the SignalAnalysis class.

        Args:
            time (array-like): Time values associated with the signals.
            force (array-like): Force signal data.
            displacement (array-like): Displacement signal data.
            input_freq (float): Input frequency for analysis.
        """
        
        self.time = time
        self.displ = displacement
        self.force = force
        self.freq = input_freq
        
    def spectral_density_signals(self, trouble_shooting = True):
        
        """
        Calculate spectral density of signals and return a frequency-related parameter.

        Args:
            trouble_shooting (bool, optional): Whether to enable troubleshooting plots. Default is True.

        Returns:
            float: The calculated frequency-related parameter.
        """
        
        shape_time = np.shape(self.time)[0]
        nfft_time = 4*shape_time
        
        # Calculate the sampling frequency
        fs = shape_time/(np.max(self.time)-np.min(self.time)) 
        
        # Compute the cross-spectral density (CSD) between displacement and force 
        [freq, PXY] = csd(self.displ, self.force, fs, nperseg = shape_time, nfft = nfft_time)
        
        # Compute the power spectral density (PSD) of displacement
        [freq, PXX] = csd(self.displ, self.displ, fs, nperseg = shape_time, nfft = nfft_time)
        
        # Calculate the frequency response function (FRF) as the ratio of CSD to PSD
        FRF = PXY/PXX
        
        # Find the frequency-related parameter from the FRF corresponding to the input frequency 
        idx = np.argmin(np.abs(freq - self.freq))
        k = FRF[idx]
        
        # Plot for trouble shooting
        if TROUBLE_SHOOTING:
            
            fig, ax = plt.subplots(figsize = (10, 6))
            
            ax.plot(freq, PXX, color = "blue", linestyle = "dashdot", label = "PXX (PSD displacement)")
            ax.plot(freq, PXY, color = "green", linestyle = "dashdot", label = "PXY (CSD displacement and force)")
            
            ax.set_xlabel("Frequency (Hz)", fontsize = 16)
            ax.set_ylabel("PXX & PXY (a.u.)", fontsize = 16)
            ax.set_xlim(0, 2 * self.freq)
            
            fig.suptitle("cross spectral density signals", fontsize = 18)
            fig.legend(bbox_to_anchor = (1, 1), loc = "upper left")
            fig.tight_layout()
            fig.show()
        
        return k
    
    def signal_amplitudes(self):
        
        """
        Calculate signal amplitudes using Fourier Transform.

        Returns:
            tuple: A tuple containing frequency and amplitude values for force and displacement.
        """
        freq_force, ampl_force = FourierTransform(self.time).signal_amplitude_fft(self.force, self.freq)
        freq_displ, ampl_displ = FourierTransform(self.time).signal_amplitude_fft(self.displ.values.tolist(), self.freq)
        
        return freq_force, ampl_force, freq_displ, ampl_displ
         
#____________________________________________________________________________________________________________________________
    
class AFMR:
    
    """
    Perform AFMR analysis on raw data and save results in csv files sorted by bead number.
    """
    def __init__(self, user_data, user_folder, results_folder, figures_folder, error_folder):
        
        """
        Initialize the AFMR class.

        Args:
            user_data (dict): Dictionary containing user's data (CSV and TDMS file dataframes).
            user_folder (str): Path to the user's folder.
            results_folder (str): Path to the folder to save analysis results.
            figures_folder (str): Path to the folder to save analysis figures.
            error_folder (str): Path to the folder to move files with errors.
        """
        
        self.data = user_data
        self.user_folder = user_folder
        self.save_results = results_folder
        self.save_figures = figures_folder
        self.error_folder = error_folder
        
    def rheology_results(self):
        
        """
        Perform rheology analysis on the provided data and save results.
        """
                   
        for filename in self.data:              
            for groupname in self.data[filename]:                
                dataframe = self.data[filename][groupname]
                
                try:
                    time = dataframe.loc[:,"time (s)"]                    
                except:
                    time = dataframe.loc[:,"time (min)"] * 60
                
                try:
                    displacement = dataframe.loc[:, "Distance (um)"] * 1000
                except:
                    displacement = dataframe.loc[:, "Distance (nm)"]
                    
                try: 
                    intensity = dataframe.loc[:, "Force (pN)"]
                except:
                    intensity = dataframe.loc[:, "Force (N)"]
                    
                UI = UserInput(groupname)
                volt = UI.voltage()
                mod = UI.modulation()
                freq = UI.frequency()
                
                force = ConvertIntensity(volt, mod).force_conversion(intensity)
                
                signal_handling = SignalAnalysis(time, force, displacement, freq)
                k = signal_handling.spectral_density_signals()
                force_freq, force_amp, displ_freq, displ_amp = signal_handling.signal_amplitudes() 
                
                # Determine whether files match input frequency
                try:
                    if any((force_freq < freq, force_freq > freq, force_freq != displ_freq, displ_amp < 20)):
                        ErrorFiles(self.user_folder, self.error_folder, filename).move_file()     
                        print(f"No match frequency for file: {filename} \n")
                        continue
                except Exception as e:
                    print(f"File {filename} error: {e}")
                    continue
                
                # Determine phase, loss tangent, sin and cos parameters                
                phase_theta = np.arctan(k.imag/k.real)                
                loss_tangent= k.imag/k.real
                sin_phase = np.sin(phase_theta)
                cos_phase = np.cos(phase_theta) 
                
                # Create a dataframe for the results to be written to csv file
                results_dict = {"File": f"{filename}{groupname}", "Frequency": freq, "Phase": phase_theta, "Loss tangent": loss_tangent, \
                                "cos(phase)": cos_phase , "sin(phase)": sin_phase, "Force amplitude (pN)": force_amp, "Displacement amplitude (nm)": displ_amp}
                results_dataframe = pd.DataFrame(results_dict, index = [0])
                
                # Initialize filepath for csv file
                bead_file = FilenameBeadResults(filename).name_file()                
                filepath = os.path.join(self.save_results, bead_file)
                
                # Write the results to a csv file
                try:
                    if not os.path.isfile(filepath):
                        results_dataframe.to_csv(filepath, sep = ";", index = False)    
    
                    else:
                        results_dataframe.to_csv(filepath, sep = ";", mode = "a", index = False, header = False)
                        
                except PermissionError:
                    print("Couldn't write data to file")
                    
                if PLOT_RAW_DATA:
                    plots = PlotData(self.save_figures, time, force, displacement, filename, groupname, freq)
                    plots.plot_raw_data()
                    plots.plot_lissajous()
                
 
#____________________________________________________________________________________________________________________________
        
# Entry point of the program
if __name__ == "__main__":
    main()