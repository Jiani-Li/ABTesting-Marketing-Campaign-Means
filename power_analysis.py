import pandas as pd
from statsmodels.stats.power import tt_ind_solve_power
from typing import Tuple


class PowerAnalysis:
    MDE: float = 0.05  # Empirical threshold
    SIGNIFICANCE_LEVEL: float = 0.05  # Industry standard
    POWER: float = 0.8  # Industry standard

    def __init__(self, data: pd.DataFrame) -> None:
        self.data: pd.DataFrame = data
        self.sd_sales: float
        self.avg_sales: float
        self.sample_size: float
        self.sd_sales, self.avg_sales = self.calculate_std_mean()
        self.sample_size = self.calculate_sample_size()

    def calculate_std_mean(self) -> Tuple[float, float]:
        sales_summary = pd.DataFrame(self.data.groupby('USER_ID')['SALES'].sum())
        sd_sales = sales_summary['SALES'].astype('float').std().round(2)
        avg_sales = sales_summary['SALES'].astype('float').mean().round(2)

        return sd_sales, avg_sales

    def calculate_sample_size(self) -> float:
        delta = self.avg_sales * self.MDE
        effect_size = delta / self.sd_sales  # cohen D: effect_size for means
        sample_size = tt_ind_solve_power(effect_size=effect_size,
                                         alpha=self.SIGNIFICANCE_LEVEL,
                                         power=self.POWER,
                                         ratio=1,
                                         alternative='two-sided', nobs1=None)

        return sample_size

    def print_analysis_result(self) -> None:
        print('------------------------------------Power Analysis------------------------------------------')
        print(f'Standard deviation of sales(sd_sales): {self.sd_sales}')
        print(f'Average sales(avg_sales): {self.avg_sales}')
        print('Sample size needed for each group of the AB test:', f'{self.sample_size:.2f}')
        print('Sample size needed for the AB test:', f'{self.sample_size * 2:.2f}')
        print('---------------------------------------------------------------------------------------------\n')