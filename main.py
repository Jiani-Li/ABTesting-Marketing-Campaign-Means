from FileInfo_and_DataReader import FilesInfo, DataReader
from PowerAnalysis import PowerAnalysis
from ABTestSetup import ABTestSetup
from ValidityChecks import ValidityChecks
from StatisticalInference import ConductStatisticalInference


# List all the files which are needed in the test
paths = ['historical_daily_sales.csv', 'historical_daily_users.csv', 'pretest_data.csv', 'abtest_result.csv']
file_info = FilesInfo(files_paths=paths)

# List all the file paths
historical_daily_sales_path = 'historical_daily_sales.csv'
historical_daily_users_path = 'historical_daily_users.csv'
pretest_data_path = 'pretest_data.csv'
abtest_data_path = 'abtest_result.csv'

# Read all the csv files
daily_sales_data = DataReader(path=historical_daily_sales_path, files_info=file_info).read_csv()
daily_users_data = DataReader(path=historical_daily_users_path, files_info=file_info).read_csv()
pretest_data = DataReader(path=pretest_data_path, files_info=file_info).read_csv()
abtest_data = DataReader(path=abtest_data_path, files_info=file_info).read_csv()

# Instances of the class
power_analysis = PowerAnalysis(data=daily_sales_data)
sample_size = power_analysis.calculate_sample_size()
ab_test_setup = ABTestSetup(sample_size=sample_size, data=daily_users_data)
validity_checks = ValidityChecks(aa_data=pretest_data, ab_data=abtest_data)
stats_inference = ConductStatisticalInference(ab_data=abtest_data)

# Print analysis results
power_analysis.print_analysis_result()
ab_test_setup.print_analysis_result()
validity_checks.print_analysis_result()
validity_checks.aa_test_plot()
stats_inference.print_analysis_result()
