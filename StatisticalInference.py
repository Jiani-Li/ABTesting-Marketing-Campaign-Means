from typing import Tuple
import pandas as pd
import statsmodels.stats.api as sms
from statsmodels.stats.weightstats import ttest_ind


class ConductStatisticalInference:
    AB_ALPHA: float = 0.05
    POWER: float = 0.8
    MDE: float = 0.05

    def __init__(self, ab_data: pd.DataFrame) -> None:
        self.ab_data: pd.DataFrame = ab_data
        self.ab_control, self.ab_treatment = self.ab_data_handler()
        self.ab_t_stat, self.ab_p_value, self.ab_df = self.analyze_statistics()
        self.lb, self.ub = self.confidence_interval_cal()

    def ab_data_handler(self) -> Tuple[pd.Series, pd.Series]:
        ab_control = self.ab_data[self.ab_data['GROUP'] == 0].groupby('USER_ID')['SALES'].mean()
        ab_treatment = self.ab_data[self.ab_data['GROUP'] == 1].groupby('USER_ID')['SALES'].mean()
        return ab_control, ab_treatment

    def analyze_statistics(self) -> Tuple[float, float, float]:
        ab_t_stat, ab_p_value, ab_df = ttest_ind(self.ab_control, self.ab_treatment)
        return ab_t_stat, ab_p_value, ab_df

    def confidence_interval_cal(self) -> Tuple[float, float]:
        desc_stats_test = sms.DescrStatsW(self.ab_treatment)
        desc_stats_control = sms.DescrStatsW(self.ab_control)
        cm = sms.CompareMeans(desc_stats_test, desc_stats_control)
        lb, ub = cm.tconfint_diff(usevar='unequal')
        return lb, ub

    def print_analysis_result(self) -> None:
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
        print('----------------------------------------------------------------------------------------------')