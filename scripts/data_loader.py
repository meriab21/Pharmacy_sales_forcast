import pandas as pd

missing_values = ["n/a", "na", 'none', "-", "--", None, '?']


def load_df_from_csv(filename: str, na_values: list = []) -> pd.DataFrame:
    """
        A simple function which tries to load a dataframe from a specified .csv filename returning the loaded DataFrame
        Parameters
        ----------
        filename:
            Type: str
        na_values:
            Type: list
            Default value = []

        Returns
        -------
        pd.DataFrame
    """
    try:
        na_values.extend(missing_values)
        df = pd.read_csv(filename, na_values=na_values)
        df = optimize_df(df)

        return df
    except:
        print("Error Occured:\n\tCould not find specified .csv file")


def load_df_from_excel(filename: str, na_values: list = []) -> pd.DataFrame:
    """
       A simple function which tries to load a dataframe from a specified .xslx filename returning the loaded DataFrame
        Parameters
        ----------
        filename:
            Type: str
        na_values:
            Type: list
            Default value = []

        Returns
        -------
        pd.DataFrame
    """
    try:
        na_values.extend(missing_values)
        df = pd.read_excel(
            filename, na_values=na_values, engine='openpyxl')
        df = optimize_df(df)

        return df
    except:
        print("Error Occured:\n\tCould not find specified .xslx file")


def optimize_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
       A simple function which optimizes the data types of the dataframe and returns it
        Parameters
        ----------
        dataframe:
            Type: pd.DataFrame

        Returns
        -------
        pd.DataFrame
    """
    data_types = dataframe.dtypes
    optimizable = ['float64', 'int64']
    for col in data_types.index:
        if(data_types[col] in optimizable):
            if(data_types[col] == 'float64'):
                # downcasting a float column
                dataframe[col] = pd.to_numeric(
                    dataframe[col], downcast='float')
            elif(data_types[col] == 'int64'):
                # downcasting an integer column
                dataframe[col] = pd.to_numeric(
                    dataframe[col], downcast='unsigned')

    return dataframe
