import pandas as pd
from typing import Dict, List


class FilesInfo:
    FILES_COLUMNS: Dict[str, List[str]] = {
        'historical_daily_sales.csv': ['DATE', 'USER_ID', 'SALES'],
        'historical_daily_users.csv': ['DATE', 'USER_ID', 'AD_SPEND'],
        'pretest_data.csv': ['USER_ID', 'DATE', 'EXPERIMENT', 'GROUP', 'SALES'],
        'abtest_result.csv': ['USER_ID', 'DATE', 'EXPERIMENT', 'GROUP', 'SALES']
    }

    def __init__(self, files_paths: List[str]) -> None:
        self.file_paths: List[str] = files_paths

    def get_file_and_columns(self, file_path: str) -> Dict[str, List[str] | None]:
        # Get the specific file and its columns
        return {file_path: self.FILES_COLUMNS.get(file_path, None)}


class DataReader:
    def __init__(self, path: str, files_info: FilesInfo) -> None:
        self.path: str = path
        self.files_info: FilesInfo = files_info
        self.csv_df: pd.DataFrame = self.read_csv()

    def read_csv(self) -> pd.DataFrame:
        file_columns = self.files_info.get_file_and_columns(self.path)
        columns = file_columns.get(self.path, None)
        if columns:
            csv_df = pd.read_csv(self.path, usecols=columns)
        else:
            csv_df = pd.read_csv(self.path)
        if 'DATE' in csv_df.columns:
            csv_df['DATE'] = pd.to_datetime(csv_df['DATE'], format='%m/%d/%y')

        return csv_df

    def print_file_info(self) -> None:
        print(self.csv_df.info())