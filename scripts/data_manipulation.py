import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, LabelEncoder
from datetime import datetime
# When importing from notebook
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
from scripts.logger_creator import CreateLogger

logger = CreateLogger('Data Manipulatior', handlers=1)
logger = logger.get_default_logger()


class DataManipulator:
    def __init__(self, df: pd.DataFrame, deep=False):
        """
            Returns a DataManipulator Object with the passed DataFrame Data set as its own DataFrame
            Parameters
            ----------
            df:
                Type: pd.DataFrame

            Returns
            -------
            None
        """
        if(deep):
            self.df = df.copy(deep=True)
        else:
            self.df = df

    def add_week_day(self, day_of_week_col: str) -> pd.DataFrame:
        try:
            date_index = self.df.columns.get_loc(day_of_week_col)
            self.df.insert(date_index + 1, 'WeekDay',
                           self.df[day_of_week_col].apply(lambda x: 1 if x <= 5 else 0))

            logger.info("Successfully Added WeekDay Column to the DataFrame")

        except Exception as e:
            logger.exception("Failed to Add WeekDay Column")

    def add_week_ends(self, day_of_week_col: str) -> pd.DataFrame:
        try:
            date_index = self.df.columns.get_loc(day_of_week_col)
            self.df.insert(date_index + 1, 'WeekEnd',
                           self.df[day_of_week_col].apply(lambda x: 1 if x > 5 else 0))

            logger.info("Successfully Added WeekEnd Column to the DataFrame")

        except Exception as e:
            logger.exception("Failed to Add WeekEnd Column")

    # Considering christmas lasts for 12 days, Easter for 50 days and public holidays for 1 day.
    # And considering before and after periods to be 5 less and 5 more days before and after the holiday for christmas
    # and 10 days for Easter
    # And 3 days for public holiday
    # get state holiday list
    # a = public holiday, b = Easter holiday, c = Christmas, 0 = None

    def affect_list(self, change_list, interval, duration, index):
        start_pt = int(index-duration/2) - interval
        try:
            for index in range(start_pt, start_pt + interval):
                change_list[index] = 'before'
            for index in range(start_pt + interval, start_pt + interval + duration):
                change_list[index] = 'during'
            for index in range(start_pt + interval + duration, start_pt + interval + duration + interval):
                change_list[index] = 'after'
        except:
            pass

        return change_list

    def modify_holiday_list(self, holiday_list: list) -> list:
        new_index = ["neither"] * len(holiday_list)
        for index, value in enumerate(holiday_list):
            if value == 'a':  # public holiday
                self.affect_list(new_index, 3, 1, index)
            elif value == 'b':  # Easter
                self.affect_list(new_index, 10, 50, index)
            elif value == 'c':  # christmas
                self.affect_list(new_index, 5, 12, index)
            else:
                pass

        return new_index

    def add_number_of_days_to_holiday(self, state_holiday_col: str):
        try:
            date_index = self.df.columns.get_loc(state_holiday_col)

            modified_index = self.modify_holiday_list(
                self.df[state_holiday_col].values.tolist())
            days_to_holiday_index = []
            i = 0
            last_holiday_index = 0
            for index, value in enumerate(modified_index):
                if(index == len(modified_index) - 1):
                    for j in range(last_holiday_index+1, len(modified_index)):
                        days_to_holiday_index.append(0)
                elif(value == 'neither' or value == 'after' or value == 'before'):
                    i += 1
                elif(value == 'during' and i != 0):
                    last_holiday_index = index
                    for j in range(i):
                        days_to_holiday_index.append(i)
                        i = i-1
                    days_to_holiday_index.append(0)
                    i = 0
                elif(value == 'during' and i == 0):
                    days_to_holiday_index.append(i)
                    last_holiday_index = index
                    continue

            self.df.insert(date_index + 1, 'DaysToHoliday',
                           days_to_holiday_index)

            logger.info("Successfully Added DaysToHoliday Column")

        except Exception as e:
            logger.exception("Failed to Add DaysToHoliday Column")

    def add_number_of_days_after_holiday(self, state_holiday_col: str):
        try:
            date_index = self.df.columns.get_loc(state_holiday_col)

            modified_index = self.modify_holiday_list(
                self.df[state_holiday_col].values.tolist())

            days_to_after_holiday_index = [0] * len(modified_index)
            i = 0
            last_holiday_index = modified_index.index('during')

            for index, value in enumerate(modified_index):
                if(value == 'before'):
                    if(index > last_holiday_index):
                        i += 1
                        days_to_after_holiday_index[index] = i
                    continue
                elif(value == 'after'):
                    i += 1
                    days_to_after_holiday_index[index] = i
                elif(value == 'during'):
                    last_holiday_index = index
                    i = 0
                    continue

            days_to_after_holiday_index.insert(0, 0)

            self.df.insert(date_index + 1, 'DaysAfterHoliday',
                           days_to_after_holiday_index[:-1])

            logger.info("Successfully Added DaysAfterHoliday Column")

        except Exception as e:
            logger.exception("Failed to Add DaysAfterHoliday Column")

    def return_day_status_in_month(self, day: int) -> int:
        # conside 1 is beginning of month, 2 is middle of the month and 3 is end of the month
        if(day <= 10):
            return 1
        elif(day > 10 and day <= 20):
            return 2
        else:
            return 3

    def add_month_timing(self, day_col: str) -> pd.DataFrame:
        try:
            date_index = self.df.columns.get_loc(day_col)
            self.df.insert(date_index + 1, 'MonthTiming',
                           self.df[day_col].apply(self.return_day_status_in_month))

            logger.info("Successfully Added MonthTiming Column")

        except Exception as e:
            logger.exception("Failed to Add MonthTiming Column")

    def get_season(self, month: int):
        if(month <= 2 or month == 12):
            return 'Winter'
        elif(month > 2 and month <= 5):
            return 'Spring'
        elif(month > 5 and month <= 8):
            return 'Summer'
        else:
            return 'Autumn'

    def add_season(self, month_col: str) -> pd.DataFrame:
        try:
            date_index = self.df.columns.get_loc(month_col)
            self.df.insert(date_index + 1, 'Season',
                           self.df[month_col].apply(self.get_season))

            logger.info("Successfully Added Season Column")

        except Exception as e:
            logger.exception("Failed to Add Season Column")

    def sort_using_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrame sorted with the specified column, default dataframe sorting
            Parameters
            ----------
            column:
                Type: str

            Returns
            -------
            pd.DataFrame
        """
        try:
            return self.df.sort_values(column)
        except:
            print("Failed to sort using the specified column")

    def get_top_sorted_by_column(self, column: str, length: int) -> pd.DataFrame:
        """
            Returns the objects DataFrame sorted in descending order and selecting the top ones with the specified column
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            pre_df = self.df.sort_values(
                column, ascending=False).iloc[:length, :]
            return pd.DataFrame(pre_df.loc[:, column])
        except:
            print("Failed to sort using the specified column and get the top results")

    def scale_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrames column scaled using MinMaxScaler
            Parameters
            ----------
            column:
                Type: str

            Returns
            -------
            pd.DataFrame
        """
        try:
            scale_column_df = pd.DataFrame(self.df[column])
            scale_column_values = scale_column_df.values
            print(
                f'The max and min values of the scaled {column} column are:\n\tmax: {scale_column_df.iloc[:, 0].min()}\n\tmin: {scale_column_df.iloc[:, 0].max()}')
            min_max_scaler = MinMaxScaler()
            scaled_values = min_max_scaler.fit_transform(scale_column_values)
            self.df[column] = scaled_values

            return self.df

        except:
            print("Failed to scale the column")

    def normalize_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            scale_column_df = pd.DataFrame(self.df[column])
            scale_column_values = scale_column_df.values
            normalizer = Normalizer()
            normalized_data = normalizer.fit_transform(scale_column_values)
            self.df[column] = normalized_data

            return self.df

        except:
            print("Failed to normalize the column")

    def standardize_column(self, column: str) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            std_column_df = pd.DataFrame(self.df[column])
            std_column_values = std_column_df.values
            standardizer = StandardScaler()
            normalized_data = standardizer.fit_transform(std_column_values)
            self.df[column] = normalized_data

            return self.df
        except:
            print("Failed to standardize the column")

    def standardize_columns(self, columns: list) -> pd.DataFrame:
        try:
            for col in columns:
                self.df = self.standardize_column(col)

            return self.df
        except:
            print(f"Failed to standardize {col} column")

    def minmax_scale_column(self, column: str, range_tup: tuple = (0, 1)) -> pd.DataFrame:
        """
            Returns the objects DataFrames column normalized using Normalizer
            Parameters
            ----------
            column:
                Type: str
            length:
                Type: int

            Returns
            -------
            pd.DataFrame
        """
        try:
            std_column_df = pd.DataFrame(self.df[column])
            std_column_values = std_column_df.values
            minmax_scaler = MinMaxScaler(feature_range=range_tup)
            normalized_data = minmax_scaler.fit_transform(std_column_values)
            self.df[column] = normalized_data

            return self.df
        except:
            print("Failed to standardize the column")

    def minmax_scale_columns(self, columns: list, range_tup: tuple = (0, 1)) -> pd.DataFrame:
        try:
            for col in columns:
                self.df = self.minmax_scale_column(col, range_tup)

            return self.df
        except:
            print(f"Failed to MinMax standardize {col} column")

    def fill_columns_with_max(self, columns: list) -> None:
        try:
            for col in columns:
                self.df[col] = self.df[col].fillna(self.df[col].max())

        except Exception as e:
            print("Failed to fill with max value")

    def fill_columns_with_most_frequent(self, columns: list) -> None:
        try:
            for col in columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])

        except Exception as e:
            print("Failed to fill with max value")

    def label_columns(self, columns: list) -> dict:
        labelers = {}
        try:
            for col in columns:
                le = LabelEncoder()
                le_fitted = le.fit(self.df[col].values)
                self.df[col] = le_fitted.transform(self.df[col].values)
                labelers[col] = le_fitted

            return labelers

        except Exception as e:
            print("Failed to Label Encode columns")

    def create_date(self, name:str='Date',columns: list=['Year','Month','Day']) -> None:
        date_column = []
        try:
            for index, row in self.df.iterrows():
                date_column.append(
                    datetime(int(row[columns[0]]), int(row[columns[1]]), int(row[columns[2]])))

            self.df[name] = date_column
        
        except Exception as e:
            print('Failed to create Date column',e)

