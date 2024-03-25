import numpy as np
import statsmodels.stats.api as sms
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Tuple, Union, Any
from pandas.core.frame import DataFrame
from pandas.core.series import Series


class ValidityChecks:
    SRM_ALPHA: float = 0.05

    def __init__(self, aa_data: DataFrame, ab_data: DataFrame) -> None:
        self.aa_data: DataFrame = aa_data
        self.ab_data: DataFrame = ab_data

        self.aa_avg_sales_per_group, self.daily_aa_avg_sales = self.aa_data_handler()
        self.lb, self.ub = self.hypothesis_test()
        self.observed, self.expected, self.chi_stats, self.p_value = self.srm_test()

    def aa_data_handler(self) -> Tuple[DataFrame, Series]:
        aa_avg_sales_per_group: DataFrame = self.aa_data.groupby(['GROUP'])['SALES'].mean().reset_index()
        daily_aa_avg_sales: Series = self.aa_data.groupby(['GROUP', 'DATE'])['SALES'].mean()

        return aa_avg_sales_per_group, daily_aa_avg_sales

    def hypothesis_test(self) -> Tuple[float, float]:
        x: Series = self.aa_data.loc[self.aa_data['GROUP'] == 0, 'SALES'].astype(float)
        y: Series = self.aa_data.loc[self.aa_data['GROUP'] == 1, 'SALES'].astype(float)
        cm = sms.CompareMeans(sms.DescrStatsW(x), sms.DescrStatsW(y))
        lb, ub = cm.tconfint_diff(usevar='unequal')

        return lb, ub

    def srm_test(self) -> Tuple[Any, float, float, float]:
        observed: Union[np.ndarray, Any] = self.ab_data.groupby('GROUP')['USER_ID'].nunique().values
        expected: float = self.ab_data['USER_ID'].nunique() * 0.5
        chi_stats, p_value = stats.chisquare(f_obs=observed, f_exp=[expected, expected])

        return observed, expected, chi_stats, p_value

    def print_analysis_result(self) -> None:
        print('-----------------------------------Validity Checks------------------------------------------')
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

    def aa_test_plot(self) -> None:
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