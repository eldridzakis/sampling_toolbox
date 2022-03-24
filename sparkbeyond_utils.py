# Special SB functions #
import re
import pandas as pd


def operational_log_number_of_features(s: str):
    """Function to extract the number of features from the operational log"""
    number_of_features = 0
    for line in s.split('\n'):
        match = re.search(r'.+(Best feature).+\) of (\d+)', line)
        if match:
            number_of_features += int(match.group(2))
    return number_of_features


def get_regression_bins(df: pd.core.frame.DataFrame):
    """Function to get the regression bins from a DP features dataframe

    Inputs
    ----------
    df    - SparkBeyond features dataframe (client.revision().features())

    Returns
    ---------
    bins  - A list of the bin values """

    # Extract the lift columns
    columns = df.columns[df.columns.str.contains('Lift')]

    # Take the first number from each column
    bins = [int(re.search(pattern="\d+", string=column).group()) for column in columns
            if re.search(pattern="\d+", string=column)]

    # Append the last number from the last column
    bins.append(int(re.search(string=columns[-1], pattern="\d+$").group()))

    return bins
