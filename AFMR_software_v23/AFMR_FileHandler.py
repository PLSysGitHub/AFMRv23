# -*- coding: utf-8 -*-
"""
Updated script created on Fri Jul 14 14:19:28 2023
Last edit: August 24, 2023

@author: Xam, Giulia, Kiki, Hannes
"""

# Import necessary modules and libraries
import os
import re
import shutil
import pandas as pd

from nptdms import TdmsFile

from tkinter import Tk
from tkinter.filedialog import askdirectory

#____________________________________________________________________________________________________________________________

class RequestFolderPath:
    """
    Requests folder path of data to analyse from the user
    Creates folders to store generated datafiles or figures
    """
    
    def __init__(self): 
        
        """
        Initialize the RequestFolderPath class.
        This constructor sets up a Tkinter GUI interface to prompt the user to choose a folder for analysis.
        """
        
        # Create Tkinter root for GUI interaction
        root = Tk()
        
        # Prompt user to choose a folder to analyse
        self.path = askdirectory()
        
        # Close Tkinter root
        root.withdraw()
        
    def folderpath(self):
        
        """
        Get the chosen folder path.

        Returns:
            str: The chosen folder path.
        """
        
        return self.path
    
    def createresultsfolder(self, foldername):
        
        """
        Create a results folder.

        Args:
            foldername (str): Name of the results folder.

        Returns:
            str: The generated folder path for results.
        """
        
        # Generate string of folder path to store datafiles with given foldername
        resultsfolderpath = os.path.join(self.path, foldername)
        
        # Create the folder, continue without raising and error if folder already exists
        try:
            os.mkdir(resultsfolderpath)
        except FileExistsError:            
            pass
        
        return resultsfolderpath
    
    def createfiguresfolder(self, foldername):
        
        """
        Create a figures folder within the chosen folder path.

        Args:
            foldername (str): The name of the subfolder to create.

        Returns:
            str: The path of the created figures folder.
        """
        
        # Generate string of folder path to store figures with given foldername
        figuresfolderpath = os.path.join(self.path, foldername)
        
        # Create the folder, continue without raising and error if folder already exists
        try:
            os.mkdir(figuresfolderpath)
        except FileExistsError:            
            pass
            
        return figuresfolderpath

#____________________________________________________________________________________________________________________________
                
class SortData:
    """
    Organizes the data from the user chosen folder in a dictionary containing the corresponding data in dataframes
    """
    
    def __init__(self, user_folder):
        
        """
        Initialize the SortData class with the user's chosen folder path.

        Args:
            user_folder (str): The path to the folder containing the data files.
        """
           
        self.folderpath = user_folder
        
        
    def csv_files_dataframes(self):
        
        """
        Read and organize CSV files from the chosen folder into dataframes.

        Returns:
            dict: A dictionary containing dataframes for each CSV file.
        """
        
        # Select csv files in the chosen folder
        csvfiles = [x for x in os.listdir(self.folderpath) if x.endswith("csv")]
        
        # Sort the csv data in a dictionary by csv filename
        csv_dataframes_dict = {}
        for csv_file in csvfiles:
            csv_file_path = os.path.join(self.folderpath, csv_file)
            
            try:
                csv_file_data = pd.read_csv(csv_file_path, sep = ";") # Read in csv file data in as a dataframe
                if csv_file_data.empty:
                    print(f"Empty file: {csv_file}")
                    continue
            except Exception as e:
                print(f"Cannot read file: {csv_file}")
                print(f"Error: {e}")
                continue
            
            #  Create a dictionary with the csv filename as key to sort the corresponding dataframe
            csv_dataframes_dict[csv_file] = csv_file_data        
        
        return csv_dataframes_dict
            
        
    def tdms_files_dataframes(self):
        
        """
        Read and organize TDMS files from the chosen folder into dataframes.

        Returns:
            dict: A dictionary containing nested dictionaries with dataframes for each TDMS file and its groups.
        """
        
        # Select tdms files in the chosen folder
        tdmsfiles = [x for x in os.listdir(self.folderpath) if x.endswith("tdms")]
        
        tdms_dataframes_dict = {}
        for tdms_file in tdmsfiles:
            tdms_file_path = os.path.join(self.folderpath, tdms_file)            
            
            try:
                tdms_file_data = TdmsFile(tdms_file_path) # Read in all tdms file data
                
                if not tdms_file_data.groups():
                    print(f"Empty file: {tdms_file}")
                    continue
            except Exception as e:
                print(f"Cannot read file: {tdms_file}")
                print(f"Error: {e}")
                continue
            
            # Retrieve groups within the tdms files, these are the datasets
            tdms_groups_dict = {}
            tdms_groups = tdms_file_data.groups()
            
            for group in tdms_groups:
                
                tdms_dict = {}
                
                # Sort the channels within the groups by channel name within a dictionary
                tdms_channels = group.channels()
                for channel in tdms_channels:
                    tdms_dict[channel.name] = channel.data
                
                # Create a dataframe of the channel data sorted in the dictionary
                tdms_dataframe = pd.DataFrame(tdms_dict) 
                tdms_groups_dict[group.name] = tdms_dataframe
                
            # Create a dictionary with the tdms filename as key to sort the corresponding dataframe
            tdms_dataframes_dict[tdms_file] = tdms_groups_dict
        
        return tdms_dataframes_dict

#____________________________________________________________________________________________________________________________
    
class UserInput:
    """
    Selects the user input information from the group name of the raw tdms data
    In our experiments the group name structure included ##V_##MD_##Hz with ## the input values of the experiment
    """
    
    def __init__(self, group_name):
        
        """
        Initialize the UserInput class with the raw tdms group name.
        
        Args:
            group_name (str): The raw tdms group name containing experiment information.
        """
        
        self.groupname = group_name
        
    def voltage(self):
        
        """
        Extract and return the voltage from the raw tdms group name.
        If extraction fails, prompt the user for input.
        
        Returns:
            float: The extracted voltage value.
        """
        
        volt_match = re.search(r'(\d+(?:\.\d+)?)V', self.groupname)
        
        try:
            input_volt = float(volt_match.group(1))
        except:
            input_volt = float(input("Input voltage: "))
        
        return input_volt
    
    def modulation(self):
        
        """
        Extract and return the modulation from the raw tdms group name.
        If extraction fails, prompt the user for input.
        
        Returns:
            float: The extracted modulation value.
        """
        
        mod_match = re.search(r'(\d+(?:\.\d+)?)MD', self.groupname)
        
        try:
            input_mod = float(mod_match.group(1))
        except:
            input_mod = float(input("Input mod: "))
            
        return input_mod
    
    def frequency(self):
        
        """
        Extract and return the frequency from the raw tdms group name.
        If extraction fails, prompt the user for input.
        
        Returns:
            float: The extracted frequency value.
        """
        
        freq_match = re.search(r'(\d+(?:\.\d+)?)Hz', self.groupname)
        
        try:
            input_freq = float(freq_match.group(1))
        except:
            input_freq = float(input("Input frequency: "))
        
        return input_freq

#____________________________________________________________________________________________________________________________

class ErrorFiles:
    """
    Moves files that cannot be analysed from the chosen folder to the error folder
    """
    
    def __init__(self, source_folder, error_folder, filename):
        
        """
        Initialize the ErrorFiles class.
        
        Args:
            source_folder (str): The source folder containing the file to be moved.
            error_folder (str): The error folder to which the file will be moved.
            filename (str): The name of the file to be moved.
        """
        
        self.source = os.path.join(source_folder, filename)
        self.error = os.path.join(error_folder, filename)
        
    def move_file(self):
        
        """
        Move the file from the source folder to the error folder.
        """
        
        shutil.move(self.source, self.error)
        

#____________________________________________________________________________________________________________________________
    
class FilenameBeadResults:
    """
    Create filename for results from tdms data per bead number
    """
    def __init__(self, raw_filename):
        
        """
        Initialize the FilenameBeadResults class.
        
        Args:
            raw_filename (str): The raw filename from which to create the new filename.
        """
        
        self.raw_filename = raw_filename
        
    def name_file(self):
        
        """
        Create a new filename based on components extracted from the raw filename.
        
        Returns:
            str: The newly constructed filename.
        """
        
        date, N_data, N_bead = re.findall(r'(\d+)_data_(\d+)_#(\d+)txyz_worked_out_data', self.raw_filename)[0]
        filename = f"{date}_data{N_data}_bead{N_bead.zfill(3)}.csv"
        
        return filename
    
#____________________________________________________________________________________________________________________________