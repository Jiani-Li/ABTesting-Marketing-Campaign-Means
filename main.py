from csv_reader import DataReader
from file_manager import ColumnsInfo, FileManager
from power_analysis import PowerAnalysis
from ab_test_setup import ABTestSetup
from validity_checks import ValidityChecks
from statistical_inference import ConductStatisticalInference

# Define file paths for the datasets
historical_daily_sales_path = 'historical_daily_sales.csv'
historical_daily_users_path = 'historical_daily_users.csv'
pretest_data_path = 'pretest_data.csv'
abtest_data_path = 'abtest_result.csv'

# Map each CSV file to its corresponding columns using the ColumnsInfo Enum
csv_file_mappings = {
    historical_daily_sales_path: ColumnsInfo.SALES_COLUMNS,
    historical_daily_users_path: ColumnsInfo.USERS_COLUMNS,
    pretest_data_path: ColumnsInfo.TEST_DATA_COLUMNS,
    abtest_data_path: ColumnsInfo.TEST_DATA_COLUMNS,
}

# Initialize FileManager with the file mappings
file_manager = FileManager(file_mappings=csv_file_mappings)

# Instantiate DataReader for each CSV file and read the data
# The DataReader uses the FileManager to get the appropriate columns for each file
daily_sales_data = DataReader(path=historical_daily_sales_path, files_info=file_manager).read_csv()
daily_users_data = DataReader(path=historical_daily_users_path, files_info=file_manager).read_csv()
pretest_data = DataReader(path=pretest_data_path, files_info=file_manager).read_csv()
abtest_data = DataReader(path=abtest_data_path, files_info=file_manager).read_csv()

# Instances of the class
power_analysis = PowerAnalysis(data=daily_sales_data)
sample_size = power_analysis.calculate_sample_size()
ab_test_setup = ABTestSetup(sample_size=sample_size, data=daily_users_data)
validity_checks = ValidityChecks(aa_data=pretest_data, ab_data=abtest_data)
stats_inference = ConductStatisticalInference(ab_data=abtest_data)


# Print analysis results
def main():
    power_analysis.print_analysis_result()
    ab_test_setup.print_analysis_result()
    validity_checks.print_analysis_result()
    validity_checks.aa_test_plot()
    stats_inference.print_analysis_result()


if __name__ == "__main__":
    main()
