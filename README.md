# Acoustic Force Micro Rheology (AFMR) analysis software (v23)

This repository contains code used for the analysis of data displayed and discussed in the paper:

Viscoelasticity of diverse biological sample quantified by Acoustic Force Microrheology (AFMR)

Giulia Bergamaschi[1, 2, 3], Kees-Karel Taris[1, 2], Andreas Biebricher[1], Xamanie Seymonson[1, 2, 4], Hannes Witt[1, 2], Erwin Peterman[1], Gijs Wuite[1, 3]

1. Vrije Universiteit Amsterdam; De Boelelaan 1105, 1081HV Amsterdam
2. Contributed to the code
3. Correspondence regarding the paper: g.bergamaschi@vu.nl (GB), g.j.l.wuite@vu.nl (GW)
4. Correspondence regarding the code: x.m.r.seymonson@vu.nl (XS)

## About

The code was developed during the project to automate the analysis of rheology data obtain from our homebuild AFS setup. The first iterations of the code were developed in 2019 (v19). 
The version used for analysis in this paper was finalized in 2022 (v22). The version published here is v22 updated for legibility and user friendliness (v23).

## Repository contents 
The software is divide in 3 python scripts in [AFMR_software_v23]():
* [AFMR_FileHandler]()
* [AFMR_RawDataHandler]()
* [AFMR_FinalResultsHandler]()
Documention on specific script functionality for classes and functions is annotated in the scripts

Data for testing the script is also included:
* [Raw AFMR data test]()

An example of complete analysis output can be found in:
* [Raw AFMR data]()
* [Results per beadnumber]()
* [No match frequency]()
* [Figures raw data]()
* [Final results]()
* [Final figures]()

## User manual
To run the code the python scripts need to be stored in the same folder on your local machine. 
To obtain the code you can clone this github repository or download it as a ZIP archive.

### Obtaining the code
#### Cloning the repository
Open a terminal on your computer and navigate to the directory where you want to store the code. 
To clone the repository run the following command

```bash
git clone https://github.com/your-username/your-repository.git
```
#### Downloading ZIP archive 
You can also download the repository from GitHub in your web browser.
Click on the big green Code button and select "Download ZIP"
When the ZIP file is saved to your computer extract the contents to your desired location.

### Installing dependencies
Before running the code use a terminal to install the code dependencies.
Navigate to your local [AFMR_software_v23]() directory using the terminal.
Run the following command to install the dependencies

```python
pip install -r AFMR_dependencies.txt
```

### Running the code
In the repository files for testing the code are included (see [Repository contents](#repository-contents)).
The analysis workflow works as follows:
#### 1. Run [AFMR_RawDataHandler]()
This opens a folder navigation GUI so you can select the folder containing raw data files. 
The input files have to be tdms file types. For testing select the folder [Raw AFMR data test]().
The output will be 3 folders within the selected folder: [Results per beadnumber](), [No match frequency]() and [Figures raw data]().
Values obtained from the analysis will be written to csv files and stored in [Results per beadnumber](). 
Data files of which the fitted frequency does not match the input frequency will be moved to [No match frequency]() and won't be included in further analysis.
Figures obtained from various plots of the raw data will be saved in [Figures raw data]()
Now the final results can be obtained.

#### 2. Run [AFMR_FinalResultsHandler]()
This open up the folder navigation GUI again. Now you select the [Results per beadnumber]() which can now be found in the [Raw AFMR data test]() folder on your computer.
The output will be 3 folders within the selected folder: [Final results]() and [Final figures]().
Values obtained from the analysis will be written to csv files and stored in [Final results]().
This files will also immediately be used to plot the final results figures which are saved in [Final figures]().

## Final remarks
* Feedback is appreciated via Github or e-mail
* Please cite the paper when using or adapting the code 
