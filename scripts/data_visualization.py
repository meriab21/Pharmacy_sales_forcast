from tkinter import Y
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Data_Viz:
    """
    Class for Data visualization of our data
    """
    def __init__(self, filehandler) -> None:
        file_handler = logging.FileHandler(filehandler)
        formatter = logging.Formatter("time: %(asctime)s, function: %(funcName)s, module: %(name)s, message: %(message)s \n")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def plot_box(self, df: pd.DataFrame, columns, color: str) -> None:
        """
        Boxplot plotting function.
        """
        fig = plt.figure(figsize=(10, 7))

        for col in columns:
            # Creating plot
            plt.boxplot(df[columns])
            plt.title(f'Plot of {col}', size=20, fontweight='bold')
            ax = plt.gca()
            ax.set_ylim(top=df[col].quantile(0.9999))
            ax.set_ylim(bottom=0)
            # show plot
            plt.show()

    def plot_box2(self, df: pd.DataFrame, col: str) -> None:
        """
        Boxplot plotting function.
        """
        plt.figure(figsize=(8, 8))
        plt.boxplot(df[col])
        plt.title(f'Plot of {col}', size=20, fontweight='bold')
        #ax = plt.gca()
        plt.show()
        logger.info("box plot successfully created")

    def plot_pie(self, df, col, title):
        """
        pie chart plotting function.
        """
        # Wedge properties
        wp = {'linewidth': 1, 'edgecolor': "blue"}

        # Creating autocpt arguments
        def func(pct, allvalues):
            absolute = int(pct / 100.*np.sum(allvalues))
            return "{:.1f}%\n({:d} g)".format(pct, absolute)

        fig, ax = plt.subplots(figsize=(10, 7))
        wedges, texts, autotexts = ax.pie(df[col[1]],
                                          autopct=lambda pct: func(
            pct, df[col[1]]),
            labels=df[col[0]].to_list(),
            startangle=90,
            wedgeprops=wp,)

        plt.setp(autotexts, size=8, weight="bold")
        ax.set_title(title)

    def plot_bar(self, x_ax, y_ax, dfs, titles, axes):
        """
        plots bar charts
        """
        for i in range(len(axes)):
            sns.barplot(x=x_ax[i], y=y_ax[i], data=dfs[i],
                        ax=axes[i]).set_title(titles[i])

        plt.show()
        logger.info("bar plot successfully created")

    def compare_binom_dist(self, count_1, count_2, sample_1, sample_2, p_1, p_2) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        xC = np.linspace(count_1-1599, count_1+1600, 3200)
        yC = scs.binom(sample_1, p_1).pmf(xC)
        ax.bar(xC, yC, alpha=0.5)
        xE = np.linspace(count_2-1599, count_2+1600, 3200)
        yE = scs.binom(sample_2, p_2).pmf(xE)
        ax.bar(xE, yE, alpha=0.5)
        plt.xlabel('Promotion')
        plt.ylabel('probability')
        plt.show()
        
        logger.info("double binomial distribution successfully created")

    def binom_distribution(self, C_aware, C_total, C_cr) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.linspace(C_aware-1599, C_aware+1600, 3200)
        y = scs.binom(C_total, C_cr).pmf(x)
        mean, var = scs.binom.stats(C_total, C_cr)
        ax.bar(x, y, alpha=0.5)
        plt.xlabel('Promotion')
        plt.ylabel('probability')
        plt.show()

        print("mean = "+str(mean))
        print("variance = "+str(var))

        logger.info("binomial distribution successfully plotted")

    def showDistribution(self, df, cols, colors):
        """
        Distribution plotting function.
        """
        for index in range(len(cols)):
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(8, 4))
            sns.displot(
                data=df, x=cols[index], color=colors[index], kde=True, height=4, aspect=2)
            plt.title(f'Distribution of ' +
                      cols[index]+' data volume', size=20, fontweight='bold')
            plt.show()

            logger.info("successfully showed distrubution")

    def summ_columns(self, df, unique=True):
        """
        shows columns and their missing values along with data types.
        """
        df2 = df.isna().sum().to_frame().reset_index()
        df2.rename(columns={'index': 'variables',
                   0: 'missing_count'}, inplace=True)
        df2['missing_percent_(%)'] = round(
            df2['missing_count']*100/df.shape[0])
        data_type_lis = df.dtypes.to_frame().reset_index()
        df2['data_type'] = data_type_lis.iloc[:, 1]

        if(unique):
            unique_val = []
            for i in range(df2.shape[0]):
                unique_val.append(len(pd.unique(df[df2.iloc[i, 0]])))
            df2['unique_values'] = pd.Series(unique_val)

        logger.info("summary successfully created")
        return df2

    def percent_missing(df: pd.DataFrame):

        # Calculate total number of cells in dataframe
        totalCells = np.product(df.shape)

        # Count number of missing values per column
        missingCount = df.isnull().sum()

        # Calculate total number of missing values
        totalMissing = missingCount.sum()

        # Calculate percentage of missing values
        print("The dataset has", round(
            ((totalMissing/totalCells) * 100), 2), "%", "missing values.")

        logger.info("succesfully displayed missing values")
