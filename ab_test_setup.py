import pandas as pd
import math
from typing import Tuple


class ABTestSetup:
    def __init__(self, sample_size: float, data: pd.DataFrame) -> None:
        self.sample_size: float = sample_size
        self.data: pd.DataFrame = data
        self.test_duration: float = self.calculate_test_duration()
        self.cost_per_user: float
        self.total_spend_needed: float
        self.daily_spend_needed: float
        self.cost_per_user, self.total_spend_needed, self.daily_spend_needed = self.calculate_spend()

    def calculate_test_duration(self) -> float:
        daily_unique_users = pd.DataFrame(self.data.groupby('DATE')['USER_ID'].nunique())
        avg_daily_unique_users = daily_unique_users.stack().mean()
        test_duration = self.sample_size * 2 / avg_daily_unique_users

        return test_duration.round(2)

    def calculate_spend(self) -> Tuple[float, float, float]:
        total_users = self.data['USER_ID'].nunique()
        total_spend = self.data['AD_SPEND'].sum()
        cost_per_user = total_spend / total_users
        total_spend_needed = self.sample_size * 2 * cost_per_user.round(2)
        daily_spend_needed = total_spend_needed / self.test_duration.round(2)
        return cost_per_user, total_spend_needed, daily_spend_needed

    def print_analysis_result(self) -> None:
        print('--------------------------------------Test Setup---------------------------------------------')
        print('Test Duration:')
        print(f'Based on the historical data, the calculated test duration: {self.test_duration} days.')
        print(f'To make the test duration to be full week, the adjusted test duration: {math.ceil(self.test_duration / 7) * 7} days({math.ceil(self.test_duration / 7)} weeks).')
        print('\nBudget needed for the test:')
        print(f'Cost per user:', f'{self.cost_per_user:.2f}')
        print(f'Total budget:', f'{self.total_spend_needed:.2f}')
        print(f'Daily budget:', f'{self.daily_spend_needed:.2f}')
        print('---------------------------------------------------------------------------------------------\n')