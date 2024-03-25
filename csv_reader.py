import pandas as pd
from file_manager import FileManager


class DataReader:
    def __init__(self, path: str, files_info: FileManager) -> None:
        self.path: str = path
        self.files_info: FileManager = files_info
        self.csv_df: pd.DataFrame = self.read_csv()

    def read_csv(self) -> pd.DataFrame:
        columns = self.files_info.get_columns(self.path)
        csv_df = pd.read_csv(self.path, usecols=columns)
        if 'DATE' in csv_df.columns:
            csv_df['DATE'] = pd.to_datetime(csv_df['DATE'], format='%m/%d/%y')

        return csv_df

    def print_df_info(self) -> None:
        print('----------------------Information of the Dataframe-----------------')
        print(self.csv_df.info())

