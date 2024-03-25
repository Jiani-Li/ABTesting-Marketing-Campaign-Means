from typing import Dict, List, Any
from enum import Enum


class ColumnsInfo(Enum):
    SALES_COLUMNS = ('DATE', 'USER_ID', 'SALES')
    USERS_COLUMNS = ('DATE', 'USER_ID', 'AD_SPEND')
    TEST_DATA_COLUMNS = ('USER_ID', 'DATE', 'EXPERIMENT', 'GROUP', 'SALES')


class FileManager:
    """
        A class to manage file mappings.

        File_mappings: A dictionary where keys are file names and values are the associated column info.
    """
    def __init__(self, file_mappings: Dict[str, Any]) -> None:
        self.file_mappings = file_mappings

    def get_columns(self, file_path: str) -> List[str]:
        info = self.file_mappings.get(file_path)
        if info:
            return list(info.value)
        else:
            raise ValueError(f'No column info available for {file_path}')

    def print_csv_columns(self, file_path: str) -> None:
        # Directly call get_columns to retrieve column information and print it.
        try:
            columns = self.get_columns(file_path)
            print('--------------------File name and columns-------------------')
            print(f'{file_path} : {columns}')
        except ValueError as e:
            print(e)
