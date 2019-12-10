# CEST Analysis
Tools to register and pre-process CEST data for use with [FABBER CEST](https://github.com/ibme-qubic/fabber_models_cest). 

You can use [conda](https://conda.io/en/latest/) to install the required dependencies. To do this run:  
```
    conda create env -n <name of conda environment> -f requirements.yml
```

## Registering Data

### Requirements:
* [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)

### Running the tool
The registration pipeline utilises FSL as its backend. This tool can perform both 2D and 3D registration of CEST, B1, and VFA T1 time-series data to a reference image. To use the tool, run the following command:
```
    python registration.py <Your-JSON-File>.json
```
The JSON file used in the regisration tool is of the form:

{  
&nbsp;&nbsp;&nbsp;&nbsp;"Analysis Path": "/Path/To/Analysis/Folder",  
&nbsp;&nbsp;&nbsp;&nbsp;"Data Path": "/Path/To/Scans/Folder",  
&nbsp;&nbsp;&nbsp;&nbsp;"Scan ID": "/Name/of/Scan/Folder(s)",  
&nbsp;&nbsp;&nbsp;&nbsp;"Reference Name": "FileName",  
&nbsp;&nbsp;&nbsp;&nbsp;"CEST Names": "FileName",  
&nbsp;&nbsp;&nbsp;&nbsp;"T1 Name": "FileName",  
&nbsp;&nbsp;&nbsp;&nbsp;"B1 Anatomical Name": "FileName",  
&nbsp;&nbsp;&nbsp;&nbsp;"B1 FA Name": "FileName",  
&nbsp;&nbsp;&nbsp;&nbsp;"Offset File Names": "FileName"  
}

The "Data Path" should point to the folder where all of the scan data is located. You can then list each individual scan within the "Scan ID" entry (can be a list). The "Analysis Path" is the folder you want the registered data to be stored.

### MT and CEST Registration

If multiple MT/CEST files need to be registered to a single Reference image, the "CEST Names" variable will take a list. Keep in mind that "Offset File Names" also needs the same number of entries as "CEST Names".  

This tool registers CEST data using the logic shown below.  
![alt text](https://github.com/smithalexk/CEST_analysis/raw/master/images/Registration.png "CEST Motion Correction Pipeline")
If there are offsets between ±1 ppm, the the registration tool with split the stack into three separate time-series of data. The > ±1 ppm data will be registered to the S<sub>0</sub> image; then a co-registered image close to 1 ppm is used as the reference to MCFLIRT the remaining volumes. The three separate time-series are then combined back into one time-series for further processing. If no offsets between ±1 ppm are detected (e.g., using MT data only), then the entire time-series will be coregistered to the S<sub>0</sub> image.

