import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def bar_plot(data, column):
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle(f'{column} Column Distribution')
    sns.countplot(x=column, data=data[data["experiment"] == "exposed"], ax=axs[0]).set_title(
        'Experiment Group')

    sns.countplot(x=column, data=data[data["experiment"]
                  == "control"], ax=axs[1]).set_title('Control Group')

def graph_comparison_bar_plot(dataframe1, dataframe2, column, plot_name):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.histplot(dataframe1[column], ax=ax1)
    sns.histplot(dataframe2[column], ax=ax2)
    fig.suptitle(f'Distribution of {plot_name}', size=20, fontweight='bold')
    # plt.show()

    return plt


def modified_bar_plot(df: pd.DataFrame, x: str, y: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df[x], y=df[y], ci=False)
    plt.title(title)
    plt.xlabel('Experiment Type')
    plt.ylabel('Response (Proportion)')


def plot_hist(df: pd.DataFrame, column: str, color: str) -> None:
    # plt.figure(figsize=(15, 10))
    # fig, ax = plt.subplots(1, figsize=(12, 7))
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(f'Distribution of user {column}', size=20, fontweight='bold')
    plt.show()


def plot_count(df: pd.DataFrame, column: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x=column)
    plt.title(f'Distribution of User {column}', size=20, fontweight='bold')
    plt.xlabel('Number of Users')
    plt.show()


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.barplot(data=df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.show()


def plot_heatmap(df: pd.DataFrame, title: str, cbar=False) -> None:
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0,
                vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)
    plt.title(title, size=18, fontweight='bold')
    plt.show()


def plot_heatmap_from_correlation(correlation, title: str):
    plt.figure(figsize=(20, 14))
    sns.heatmap(correlation)
    plt.title(title, size=18, fontweight='bold')
    plt.show()


def plot_box(df: pd.DataFrame, x_col: str, title: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x=x_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.show()


def plot_box_multi(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def simple_plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(title, size=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
