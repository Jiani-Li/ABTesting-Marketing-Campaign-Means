# AB Testing Marketing Campaign(Means):Sponsored Search Ranking Algorithm Experimentation

## Description
This project outlines an experimental framework conducted by Amazon's search team to assess the effectiveness of a new sponsored search ranking algorithm. The primary objective of this experiment is to enhance the precision of product suggestions for customers, thereby potentially increasing customer satisfaction and sales metrics.
## Code Structure
The codebase is organized into several Python modules, each responsible for handling specific aspects of the experiment. Below is a brief overview of each module and its role within the project:

## FileInfo_and_DataReader.py
### Classes: 
FilesInfo, DataReader
### Purpose: 
Manages file paths for the experiment's datasets and provides functionality to read these files into pandas DataFrame objects. Ensures only the necessary columns are imported to optimize memory usage.
## PowerAnalysis.py
### Class: PowerAnalysis
### Purpose: 
Conducts a power analysis to determine the required sample size for the experiment, ensuring that the experiment is adequately powered to detect a statistically significant difference between the control and treatment groups.
## ABTestSetup.py
### Class: ABTestSetup
### Purpose: 
Configures the A/B test setup, including calculation of the test duration and budgeting based on historical data and the calculated sample size.
## ValidityChecks.py
### Class: ValidityChecks
### Purpose: 
Performs validity checks on the experiment data, including an AA test to ensure there are no inherent biases in the experimental setup and an SRM (Sample Ratio Mismatch) test to confirm the equal distribution of participants across control and treatment groups.
## StatisticalInference.py
### Class: ConductStatisticalInference
### Purpose: 
Conducts the statistical analysis on the experiment results to evaluate the effectiveness of the new algorithm. This includes comparing key metrics between the control and treatment groups to determine if there are significant improvements.
## Workflow
### Data Preparation: 
The FilesInfo and DataReader classes are used to list and read the necessary files, including historical sales and user data, pretest data, and A/B test results.

### Power Analysis: 
The PowerAnalysis class calculates the sample size needed for the experiment based on the historical sales data.

### Experiment Setup: 
The ABTestSetup class uses the sample size and historical user data to determine the test duration and budget requirements.

### Validity Checks: 
Before analyzing the A/B test results, the ValidityChecks class performs pre-analysis checks to ensure the integrity of the experimental setup.

### Statistical Analysis: 
Finally, the ConductStatisticalInference class analyzes the A/B test results to evaluate the effectiveness of the new sponsored search ranking algorithm.
