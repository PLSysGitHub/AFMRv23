# -*- coding: utf-8 -*-
"""
Updated script created on Fri Jul 14 14:26:50 2023
Last edit: August 24, 2023

@author: Xam, Giulia, Kiki, Hannes
"""

# Import necessary modules and libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import sem
from scipy.optimize import curve_fit

from AFMR_FileHandler import RequestFolderPath, SortData

#____________________________________________________________________________________________________________________________

# Variables used in calculating the viscous and elastic moduli
BEAD_RADIUS_GELS = 6.39 / 2
PREFACTOR_GELS = 1 / (6 * np.pi * BEAD_RADIUS_GELS)

BEAD_RADIUS_CELLS = 6.59 / 2
CELL_CORRECTION_ANGLE = np.pi / 6 # 30 degrees (np.sin(x) uses x in radians)
CELL_CORRECTION_FACTOR = ( 9 / (4 * np.sin(CELL_CORRECTION_ANGLE)) ) + ( (3 * np.cos(CELL_CORRECTION_ANGLE)) / (2 * np.sin(CELL_CORRECTION_ANGLE)**3 ) )
PREFACTOR_CELLS = CELL_CORRECTION_ANGLE / (6 * np.pi * BEAD_RADIUS_CELLS)

cm = 1/2.54 # Used to convert figsize values to centimeters

#____________________________________________________________________________________________________________________________

def main():
    
    """
    Main function to perform analysis and plotting on user data.
    """
    
    # Initialize GUI which requests the folder path from the user
    request_folder = RequestFolderPath()
    user_folder = request_folder.folderpath()
    
    # Create folder for storing result files and figures 
    results_foldername = "Final Results"
    results_folder = request_folder.createresultsfolder(results_foldername)
    
    figures_foldername = "Final Figures"
    figures_folder = request_folder.createfiguresfolder(figures_foldername)
    
    # Retrieve data from the csv files within the chosen folder and group together
    csv_data = SortData(user_folder).csv_files_dataframes()        
    all_data = pd.concat(csv_data.values(), ignore_index=True)
    
    # Perform final step of AFMR analysis on the pre-processed data
    final_results = FinalResultsAnalysis(all_data, results_folder)
    all_results = final_results.combined_results(PREFACTOR_GELS)
    avg_results = final_results.average_results()
    
    # Generate plots of the final results and fit data
    final_plots = ResultsPlotting(all_results, avg_results, figures_folder)
    final_plots.fitteddata()
    final_plots.plotdata()
    
    
#____________________________________________________________________________________________________________________________

class FinalResultsAnalysis:
    
    """
    Perform final step of analysis on provided data and save results.
    """
    
    def __init__(self, user_data, results_folder):
        
        """
        Initialize the FinalResultsAnalysis class.

        Args:
            user_data (DataFrame): DataFrame containing user's analysis results.
            results_folder (str): Path to the folder to save final results.
        """
        
        self.data = user_data
        self.save_results = results_folder
        
    def combined_results(self, prefactor, results_folder_name = "All final results.csv"):
        
        """
        Combine and calculate final analysis results with a prefactor.

        Args:
            prefactor (float): Prefactor value for calculations.
            results_folder_name (str, optional): Name of the results CSV file. Defaults to "All final results.csv".

        Returns:
            DataFrame: Combined final analysis results.
        """
        
        # Initialize the results dataframe
        column_names = ["File", "Frequency", "Phase", "Loss tangent", "Viscous component", "Elastic component"]
        self.results_data = pd.DataFrame(columns = column_names)
        
        for index, row in self.data.iterrows():
            amplitude_ratio = row["Force amplitude (pN)"] / row["Displacement amplitude (nm)"]
            elastic_component = prefactor * amplitude_ratio * row["cos(phase)"]
            viscous_component = prefactor * amplitude_ratio * row["sin(phase)"]
            
            # Add the list with calculated results for the current row to the results dataframe
            results = [row["File"], row["Frequency"], row["Phase"], row["Loss tangent"], viscous_component, elastic_component]
            self.results_data.loc[len(self.results_data)] = results
        
        # Initialize filepath for csv file of all the results
        filepath = os.path.join(self.save_results, results_folder_name)
        
        # Write all the results to a csv file
        try:
            if not os.path.isfile(filepath):
                self.results_data.to_csv(filepath, sep = ";", index = False)    
    
            else:
                pass
        except PermissionError as e:
            print(f"Couldn't write to file: {e}")
            
        return self.results_data
    
    def average_results(self, results_folder_name = "Avg final results.csv"):
        
        """
        Calculate average results from the combined analysis results.

        Args:
            results_folder_name (str, optional): Name of the results CSV file. Defaults to "Avg final results.csv".

        Returns:
            DataFrame: Averaged and sorted analysis results.
        """
        
        avg_data = {}
        
        phase_dict = {}
        loss_tangent_dict = {}
        viscous_comp_dict = {}
        elastic_comp_dict = {}
        
        # Populate dictionaries with values based on frequency
        for index, row in self.results_data.iterrows():
            
            phase_dict.setdefault(row["Frequency"], []).append(row["Phase"])
            loss_tangent_dict.setdefault(row["Frequency"], []).append(row["Loss tangent"])
            viscous_comp_dict.setdefault(row["Frequency"], []).append(row["Viscous component"])
            elastic_comp_dict.setdefault(row["Frequency"], []).append(row["Elastic component"])
        
        # Calculate average values and standard errors of the mean (sem) and store in dictionary
        for key in phase_dict:
                        
            avg_phase = np.mean(phase_dict[key])
            sem_phase = sem(phase_dict[key])
            
            avg_loss_tangent = np.mean(loss_tangent_dict[key])
            sem_loss_tangent = sem(loss_tangent_dict[key])
            
            avg_viscous_comp = np.mean(viscous_comp_dict[key])
            sem_viscous_comp = sem(viscous_comp_dict[key])
            
            avg_elastic_comp = np.mean(elastic_comp_dict[key])
            sem_elastic_comp = sem(elastic_comp_dict[key])
            
            avg_data.setdefault("Frequency", []).append(key)
            
            avg_data.setdefault("Avg Phase", []).append(avg_phase)
            avg_data.setdefault("sem(Phase)", []).append(sem_phase)
            
            avg_data.setdefault("Avg Loss tangent", []).append(avg_loss_tangent)
            avg_data.setdefault("sem(Loss tangent)", []).append(sem_loss_tangent)
            
            avg_data.setdefault("Avg Viscous component", []).append(avg_viscous_comp)
            avg_data.setdefault("sem(Viscous component)", []).append(sem_viscous_comp)
            
            avg_data.setdefault("Avg Elastic component", []).append(avg_elastic_comp)
            avg_data.setdefault("sem(Elastic component)", []).append(sem_elastic_comp)
       
        # Write dictionary of average results to dataframe and sort based on frequency value
        avg_results = pd.DataFrame(avg_data)
        sorted_results = avg_results.sort_values(by = "Frequency").reset_index(drop = True)
        
        # Initialize filepath for csv file of the average results
        filepath = os.path.join(self.save_results, results_folder_name)
        
        # Write the sorted average results to a csv file
        try:
            if not os.path.isfile(filepath):
                sorted_results.to_csv(filepath, sep = ";", index = False)    
            else:                
                pass                
        except PermissionError as e:
            print(f"Couldn't write data to file: {e}")
       
        return sorted_results

#____________________________________________________________________________________________________________________________

class ResultsPlotting:
    
    """
    Plot and save final AFMR results figures.
    """
    
    def __init__(self, fulldataframe, avgdataframe, figures_folder):
        
        """
        Initialize the ResultsPlotting class.

        Args:
            fulldataframe (DataFrame): DataFrame containing all data for plotting.
            avgdataframe (DataFrame): DataFrame containing averaged data for plotting.
            figures_folder (str): Path to the folder to save the figures.
        """
        
        self.save_figures = figures_folder
        self.all_data = fulldataframe
        self.avg_data = avgdataframe
        
    def fitteddata(self):
        
        """
        Fit data and calculate scaling parameters for the plot.
        This method fits data to a custom exponential function and calculates scaling parameters.
        """
        
        # Select data within the specified frequency range
        self.fit_data = self.avg_data[(self.avg_data["Frequency"] >= 0.3) & (self.avg_data["Frequency"] <= 100)]
        
        # Fit data to the exponential function using curve_fit
        popt_vis, pcov_vis = curve_fit(myExpFunc, self.fit_data["Frequency"], self.fit_data["Avg Viscous component"], sigma = self.fit_data["sem(Viscous component)"])
        self.vis_scale = popt_vis[1]
        self.perr_vis = np.sqrt(np.diag(pcov_vis))[1]
        self.viscous_fit = myExpFunc(self.fit_data["Frequency"], *popt_vis)
        
        popt_el, pcov_el = curve_fit(myExpFunc, self.fit_data["Frequency"], self.fit_data["Avg Elastic component"], sigma = self.fit_data["sem(Elastic component)"])
        self.el_scale = popt_el[1]
        self.perr_el = np.sqrt(np.diag(pcov_el))[1]
        self.elastic_fit = myExpFunc(self.fit_data["Frequency"], *popt_el)
        
    def plotdata(self):
        
        """
        Plot and save figures.
        """
        
        # Generate random noise for simulating data variability
        sigma = 0.08
        mu = 1
        noise = sigma * np.random.randn(len(self.all_data.loc[:, "Frequency"])) + mu
        
        noisey_frequency = self.all_data.loc[:, "Frequency"] * noise
        avg_frequency = self.avg_data.loc[:, "Frequency"] 
        
        # Create the Phase delay figure
        fig1, ax1 = plt.subplots(figsize = (8 * cm, 5.5 * cm))
        
        ax1.scatter(noisey_frequency, self.all_data.loc[:, "Phase"], edgecolor = "orange", facecolor = "none", alpha = 0.5)
        ax1.scatter(avg_frequency, self.avg_data.loc[:, "Avg Phase"], color = "darkorange")
        ax1.set_xlabel("Frequency (Hz)", fontsize = 14)
        ax1.set_ylabel("Phase", fontsize = 14)
        ax1.set_xscale("log")
        ax1.set_yscale("log")        
        
        fig1name = "Phase delay.pdf"
        fig1.savefig(os.path.join(self.save_figures, fig1name), bbox_inches = "tight")
        
        fig1.show()
        
        # Create the Loss tangent figure
        fig2, ax2 = plt.subplots(figsize = (8 * cm, 5.5 * cm))
        
        ax2.scatter(noisey_frequency, self.all_data.loc[:, "Loss tangent"], edgecolor = "hotpink", facecolor = "none", alpha = 0.5)
        ax2.scatter(avg_frequency, self.avg_data.loc[:, "Avg Loss tangent"], color = "deeppink")
        ax2.set_xlabel("Frequency (Hz)", fontsize = 14)
        ax2.set_ylabel("Loss tangent",  fontsize = 14)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        
        fig2name = "Loss tangent.pdf"
        fig2.savefig(os.path.join(self.save_figures, fig2name), bbox_inches = "tight")
        
        fig2.show()
        
        # Create the Rheology results figure
        fig3, ax3 = plt.subplots(figsize = (8 * cm, 5.5 * cm))
        
        ax3.scatter(noisey_frequency, self.all_data.loc[:, "Viscous component"], edgecolor = "cornflowerblue", facecolor = "white", alpha = 0.25)
        ax3.errorbar(avg_frequency, self.avg_data.loc[:, "Avg Viscous component"], self.avg_data.loc[:, "sem(Viscous component)"], markersize = 4, capsize = 2, fmt = "o", ecolor = "royalblue", elinewidth = 2, markeredgecolor = "royalblue", markerfacecolor = "white", label = "Viscous modulus (G'')", zorder=1)
        ax3.plot(self.fit_data["Frequency"], self.viscous_fit, color = "turquoise", linestyle = "dashdot", label = u"PL scaling: {0:.2f} \u00B1 {1:.2f}".format(self.vis_scale, self.perr_vis))
        
        ax3.scatter(noisey_frequency, self.all_data.loc[:, "Elastic component"], color = "navy", alpha = 0.25)
        ax3.errorbar(avg_frequency, self.avg_data.loc[:, "Avg Elastic component"], self.avg_data.loc[:, "sem(Elastic component)"], markersize = 4, capsize = 2, fmt = "o", ecolor = "midnightblue", elinewidth = 2, color = "midnightblue", label = "Elastic modulus (G')", zorder=1)
        ax3.plot(self.fit_data["Frequency"], self.elastic_fit, color = "blue", linestyle = "dashdot", label = u"PL scaling: {0:.2f} \u00B1 {1:.2f}".format(self.el_scale, self.perr_el))
        
        ax3.set_xlabel("Frequency (Hz)", fontsize = 14)
        ax3.set_ylabel("G' and G'' (Pa)", fontsize = 14)
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        fig3.legend(bbox_to_anchor = (1, 1), loc = "upper left")
        
        fig3name = "Rheology results.pdf"
        fig3.savefig(os.path.join(self.save_figures, fig3name), bbox_inches = "tight")
        
        fig3.show()
        
#____________________________________________________________________________________________________________________________
    
def myExpFunc(x, a, b):
    
    """
    Custom exponential function used to fit viscous and elastic moduli.

    Args:
        x (float): Input value.
        a (float): Parameter.
        b (float): Parameter.

    Returns:
        float: The calculated value.
    """
    
    return a * np.power(x, b)        
       
#____________________________________________________________________________________________________________________________

# Entry point of the program
if __name__ == "__main__":
    main()