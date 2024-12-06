# ELISA plate analysis tool

This tool is designed to help analyze ELISA plate data. It is designed to be used with the output from a plate reader, which typically includes the raw absorbance values for each well in a 96-well plate. The tool will calculate the average absorbance for each sample, subtract the average absorbance of the blank wells, and plot the results. It also works with data from multiple plates, allowing you to compare results across different experiments or averages of replicates from different plates.

## Usage
Copy in Excel the 96 well plate data from the plate reader. The data should be exactly 8 rows by 12 columns, if only part of the plate was used, still include empty wells. 

Annotate the data either manually or automatically using the tool. The tool will calculate averages per sample.

User can also normalize the data to a control sample / value.

Plotting can be done of the averages along the plates or within a single plate.

Final data can be downloaded as a CSV file.