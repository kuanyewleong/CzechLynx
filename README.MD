# A Tool to Recognize Individual Lyxn using AI Model 

## Download Dataset
Get the zipped data file from here:
```
https://www.kaggle.com/code/picekl/czechlynx-baseline-with-wildfusion
```
Next, move the downloaded file to your VSCode workspace.
And unzip it using this line (for Windows):
```
Expand-Archive -Path .\archive.zip -DestinationPath .
```

Create an empty folder "data", and move all the unzipped contents (the dataset) into "data"


## Technical Setup Instructions for Visualizing the Data Samples
Create a virtual environment
```
python --version
python -m venv .venv
```
Activate the virtual environment
```
.venv\Scripts\Activate.ps1
```

Install the needed python packages
```
pip install -r requirements.txt
```


## CzechLynx baseline with WildFusion (preparing the recognition model)
Install the needed python packages
```
pip install git+https://github.com/WildlifeDatasets/wildlife-datasets@develop
pip install git+https://github.com/WildlifeDatasets/wildlife-tools
```
