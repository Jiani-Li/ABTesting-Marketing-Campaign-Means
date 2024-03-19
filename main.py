import pandas as pd
from statsmodels.stats.power import ttest_power, tt_ind_solve_power
from statsmodels.stats.weightstats import ttest_ind
import statsmodels.api as sm
import statsmodels.stats.api as sms
import scipy.stats as stats
import matplotlib.pyplot as plt
import math


class DataReader:
    def __init__(self, path):
        self.path = path
        self.csv_df = self.read_csv()

    def read_csv(self):
        csv_df = pd.read_csv(self.path, delimiter=',')
        csv_df['DATE'] = pd.to_datetime(csv_df['DATE'], format='%m/%d/%y')

        return csv_df

    def print_file_info(self):
        print(self.read_csv().info)


class PowerAnalysis:
    # Power Analysis Input Parameters
    MDE = 0.05  # Empirical threshold
    SIGNIFICANCE_LEVEL = 0.05  # Industry standard
    POWER = 0.8  # Industry standard

    def __init__(self, data):
        self.data = data
        self.sd_sales, self.avg_sales = self.calculate_std_mean()
        self.sample_size = self.calculate_sample_size()

    def calculate_std_mean(self):
        sales_summary = pd.DataFrame(self.data.groupby('USER_ID')['SALES'].sum())
        sd_sales = sales_summary['SALES'].astype('float').std().round(2)
        avg_sales = sales_summary['SALES'].astype('float').mean().round(2)

        return sd_sales, avg_sales

    def calculate_sample_size(self):
        delta = self.avg_sales * self.MDE
        effect_size = delta / self.sd_sales  # cohen D: effect_size for means
        # Calculate for equal sample size, output is for one sample size
        sample_size = tt_ind_solve_power(effect_size=effect_size,
                                         alpha=self.SIGNIFICANCE_LEVEL,
                                         power=self.POWER,
                                         ratio=1,
                                         alternative='two-sided', nobs1=None)

        return sample_size

    def print_analysis_result(self):
        print('------------------------------------Power Analysis------------------------------------------')
        print(f'Standard deviation of sales(sd_sales): {self.sd_sales}')
        print(f'Average sales(avg_sales): {self.avg_sales}')
        print('Sample size needed for each group of the AB test:', f'{self.sample_size:.2f}')
        print('Sample size needed for the AB test:', f'{self.sample_size * 2:.2f}')
        print('---------------------------------------------------------------------------------------------\n')


class TestSetup:
    def __init__(self, sample_size, data):
        self.sample_size = sample_size
        self.data = data
        self.test_duration = self.calculate_test_duration()
        self.cost_per_user, self.total_spend_needed, self.daily_spend_needed = self.calculate_spend()

    def calculate_test_duration(self):
        daily_unique_users = pd.DataFrame(self.data.groupby('DATE')['USER_ID'].nunique())
        avg_daily_unique_users = daily_unique_users.stack().mean()
        # Calculate test duration based on sample size
        test_duration = self.sample_size * 2 / avg_daily_unique_users

        return test_duration.round(2)

    def calculate_spend(self):
        total_users = self.data['USER_ID'].nunique()
        total_spend = self.data['AD_SPEND'].sum()
        cost_per_user = total_spend / total_users
        total_spend_needed = self.sample_size * 2 * cost_per_user.round(2)
        daily_spend_needed = total_spend_needed / self.test_duration.round(2)
        return cost_per_user, total_spend_needed, daily_spend_needed

    def print_analysis_result(self):
        print('--------------------------------------Test Setup---------------------------------------------')
        print('Test Duration:')
        print(f'Based on the historical data, the calculated test duration: {self.test_duration} days.')
        # We usually want to make the test duration to be full weeks
        print(
            f'To make the test duration to be full week, the adjusted test duration: {math.ceil(self.test_duration / 7) * 7} days({math.ceil(self.test_duration / 7)} weeks).')
        print('\nBudget needed for the test:')
        print(f'Cost per user:', f'{self.cost_per_user:.2f}')
        print(f'Total budget:', f'{self.total_spend_needed:.2f}')
        print(f'Daily budget:', f'{self.daily_spend_needed:.2f}')
        print('---------------------------------------------------------------------------------------------\n')


class ValidityChecks:
    # Set test parameter
    SRM_ALPHA = 0.05

    def __init__(self, aa_data, ab_data):
        self.aa_data = aa_data
        self.ab_data = ab_data

        self.aa_avg_sales_per_group, self.daily_aa_avg_sales = self.aa_data_handler()
        self.lb, self.ub = self.hypothesis_test()
        self.observed, self.expected, self.chi_stats, self.p_value = self.srm_test()

    def aa_data_handler(self):
        aa_avg_sales_per_group = self.aa_data.groupby(['GROUP'])['SALES'].mean().reset_index()
        daily_aa_avg_sales = self.aa_data.groupby(['GROUP', 'DATE'])['SALES'].mean()

        return aa_avg_sales_per_group, daily_aa_avg_sales

    def hypothesis_test(self):
        x = self.aa_data.loc[self.aa_data['GROUP'] == 0, 'SALES'].astype(float)
        y = self.aa_data.loc[self.aa_data['GROUP'] == 1, 'SALES'].astype(float)
        # Calculate Confidence Interval of the means
        cm = sms.CompareMeans(sms.DescrStatsW(x), sms.DescrStatsW(y))
        lb, ub = cm.tconfint_diff(usevar='unequal')

        return lb, ub

    def srm_test(self):
        # Get the observed and expected counts in the experiment
        observed = self.ab_data.groupby('GROUP')['USER_ID'].nunique().values
        expected = self.ab_data['USER_ID'].nunique() * 0.5  # Half of total test users
        # Perform Chi-Square Goodness of Fit Test
        chi_stats, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

        return observed, expected, chi_stats, p_value

    def print_analysis_result(self):
        print('-----------------------------------Validity Checks-------------------------------------------')
        print('AA Test:')
        print('Group by the AA test groups, and calculate the average sales for each group:')
        print(self.aa_avg_sales_per_group)
        print(
            f'Lower Bound of the Confidence Interval: {float(self.lb):.03f}\nUpper Bound of the Confidence Interval: {float(self.ub):.03f}')
        if 0 > float(self.lb) and 0 < float(self.ub):
            print('Based on the AA test, the two samples are not statistically different.')
        else:
            print('Based on the AA test, the two samples are statistically different')
        print('\nSRM Test (Sample Ratio Mismatch):')
        print("Observed Counts of each group:", self.observed)
        print("Expected Count of each group:", self.expected)
        print('Ho: The ratio of samples is 1:1.')
        print('Ha: The ratio of samples is not 1:1.')
        print(f'Significance level: {self.SRM_ALPHA}')
        print(f'Chi-Square = {self.chi_stats:.3f} | P-value = {self.p_value:.3f}')
        print('Conclusion:')
        if self.p_value < self.SRM_ALPHA:
            print(
                'Reject Ho and conclude that there is statistical significance in the ratio\nof samples not being 1:1. Therefore, there is SRM.')
        else:
            print('Fail to reject Ho. Therefore, there is no SRM.')
        print('--------------------------------------------------------------------------------------------\n')

    def aa_test_plot(self):
        # Average sales per user per day
        aa_ctrl_avg_sales = self.daily_aa_avg_sales.loc[0]
        aa_trt_avg_sales = self.daily_aa_avg_sales.loc[1]

        # Get the day range of experiment
        exp_days = range(1, self.aa_data['DATE'].nunique() + 1)

        # Display the avg sales per experiment day
        f, ax = plt.subplots(figsize=(10, 6))
        # Generate plots
        ax.plot(exp_days, aa_ctrl_avg_sales, label='Control', color='b')
        ax.plot(exp_days, aa_trt_avg_sales, label='Treatment', color='g')

        # Format plot
        ax.set_xticks(exp_days)
        ax.set_title('AA Test')
        ax.set_ylabel('Avg Sales per Day')
        ax.set_xlabel('Days in the Experiment')
        ax.legend()
        plt.show()


class ConductStatisticalInference:
    # Set the ALPHA for the AB test
    AB_ALPHA = 0.05
    # Set the statistical power
    POWER = 0.8
    # Set the Minimum Detectable Effect(MDE)
    MDE = 0.05

    def __init__(self, ab_data):
        self.ab_data = ab_data
        self.ab_control, self.ab_treatment = self.ab_data_handler()
        self.ab_t_stat, self.ab_p_value, self.ab_df = self.analyze_statistics()
        self.lb, self.ub = self.confidence_interval_cal()

    def ab_data_handler(self):
        # Grab the control and treatment avg_sales_per_user in the AB test
        ab_control = self.ab_data[self.ab_data['GROUP'] == 0].groupby('USER_ID')['SALES'].mean()
        ab_treatment = self.ab_data[self.ab_data['GROUP'] == 1].groupby('USER_ID')['SALES'].mean()

        return ab_control, ab_treatment

    def analyze_statistics(self):
        # Analyze Statistics including t-statistic, p-value and degrees of freedom
        ab_t_stat, ab_p_value, ab_df = ttest_ind(self.ab_control, self.ab_treatment)

        return ab_t_stat, ab_p_value, ab_df

    def confidence_interval_cal(self):
        # Create two descriptive statistics objects using test and control data
        desc_stats_test = sm.stats.DescrStatsW(self.ab_treatment)
        desc_stats_control = sm.stats.DescrStatsW(self.ab_control)
        # Compare the means of the two datasets
        cm = sms.CompareMeans(desc_stats_test, desc_stats_control)
        # Calculate the confidence interval for the difference between the means (using unequal variances)
        lb, ub = cm.tconfint_diff(usevar='unequal')

        return lb, ub

    def print_analysis_result(self):
        control_sales = self.ab_control.sum()
        treatment_sales = self.ab_treatment.sum()
        control_avg_sales = self.ab_control.mean()
        treatment_avg_sales = self.ab_treatment.mean()
        control_size = self.ab_control.count()
        treatment_size = self.ab_treatment.count()
        absolute_diff = treatment_avg_sales - control_avg_sales
        relative_lift = (treatment_avg_sales - control_avg_sales) / control_avg_sales * 100
        lower_lift = self.lb / control_avg_sales
        upper_lift = self.ub / control_avg_sales
        print('-----------------------------Conduct Statistical Inference----------------------------------')
        print('Ho: The average attributed sales per user between 2 search ranking algorithms are the same.')
        print('H1: The average attributed sales per user between 2 search ranking algorithms are different.')
        print(f'Significance level: {self.AB_ALPHA} | Statistical Power: {self.POWER} | MDE: {self.MDE}')
        print('\nIndependent Samples t-test results:')
        print(
            f'T-Statistic: {self.ab_t_stat:.3f} | P-value: {self.ab_p_value:.3f} | Degrees of freedom: {self.ab_df:.3f}')
        print('\nSample size:')
        print(f'Control: {control_size} | Treatment: {treatment_size}')
        print('Group Statistical data:')
        print(f'Control: Total sales is {control_sales:.3f}  | Average sales is {control_avg_sales:.3f}')
        print(f'Treatment: Total sales is {treatment_sales:.3f}  | Average sales is {treatment_avg_sales:.3f}')
        print('Differences:')
        print(f'Absolute: {absolute_diff:.3f} | Relative (lift): {relative_lift:.3f}%')
        print('\nConfidence Interval:')
        print(f'Absolute Difference CI: ({self.lb:.3f}, {self.ub:.3f})')
        print(f'Relative Difference (lift) CI: ({lower_lift * 100:.3f}%, {upper_lift * 100:.3f}%)')
        print('---------------------------------------------------------------------------------------------')


historical_daily_sales_path = 'historical_daily_sales.csv'
historical_daily_users_path = 'historical_daily_users.csv'
pretest_data_path = 'pretest_data.csv'
abtest_data_path = 'abtest_result.csv'
daily_sales_data = DataReader(historical_daily_sales_path).read_csv()
daily_users_data = DataReader(historical_daily_users_path).read_csv()
pretest_data = DataReader(pretest_data_path).read_csv()
abtest_data = DataReader(abtest_data_path).read_csv()
PowerAnalysis(daily_sales_data).print_analysis_result()
sample_size = PowerAnalysis(daily_sales_data).calculate_sample_size()
TestSetup(sample_size, daily_users_data).print_analysis_result()
ValidityChecks(pretest_data, abtest_data).print_analysis_result()
ValidityChecks(pretest_data, abtest_data).aa_test_plot()
ConductStatisticalInference(abtest_data).print_analysis_result()
