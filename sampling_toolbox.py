import pandas as pd
import numpy as np
import re

# Functions for calculating metrics


def entropy(x: float):
    """Function for returning entropy"""
    if (x == 0) | (x == 1):
        return 0
    else:
        h = (-x*np.log2(x)) - ((1-x)*np.log2(1-x))
        return h


def rig(df: pd.DataFrame, target: str, feature: str):
    """Function for calculating the relative information gain"""

    h_prior = entropy(df[target].mean())
    
    # Probability feature is true
    probability_feature_true = df[feature].mean()

    # Entropy of target given feature
    s_entropy = pd.crosstab(df[feature], df[target], normalize='index').loc[:, 1].apply(entropy)

    # If the feature only has one outcome
    if len(s_entropy) < 2:
        if s_entropy.index[0] == 0:
            h_feature_F = s_entropy[0]*(1-probability_feature_true)
            h_feature_T = 0
        else:
            h_feature_F = 0
            h_feature_T = s_entropy[1]*probability_feature_true
        
    else:
        h_feature_T = s_entropy[1]*probability_feature_true
        h_feature_F = s_entropy[0]*(1-probability_feature_true)

    return (h_prior - (h_feature_T + h_feature_F))/h_prior


def rig_all_columns(df: pd.DataFrame, target: str):
    columns = df.columns.tolist()
    
    columns.remove(target)
    
    rigs = []
    for column in columns:
        rigs.append(rig(df, target, column))
        
    return rigs


def calculate_null_rigs(df: pd.DataFrame, target: str, resamples: int):
    df_shuffled = df.copy()
    null_rigs = []

    for n in range(resamples):
        df_shuffled[target] = df_shuffled[target].sample(frac=1).values

        null_rigs.append(rig_all_columns(df_shuffled, target))

    columns = df.columns.tolist()
    columns.remove(target)

    df_null_rigs = pd.DataFrame(null_rigs, columns=columns)
    
    return df_null_rigs


# Special SB functions #


def operational_log_number_of_features(s: str):
    """Function to extract the number of features from the operational log"""
    number_of_features = 0
    for line in s.split('\n'):
        match = re.search(r'.+(Best feature).+\) of (\d+)', line)
        if match:
            number_of_features += int(match.group(2))
    return number_of_features

# Main permutation code #


class PermutationObject:
    
    def __init__(self):
        
        # Default Feature Fractions (feature support)
        self.feature_fractions = [0.1]
        
        # Default reference information gain level
        self.gain_threshold = 0.0005

        # Other attributes that will be set later
        self.nrows = np.nan
        self.minority_class = np.nan
        self.data = pd.DataFrame()
        self.null_rigs = []
        self.null_rig_values = pd.Series()
        self.permutations = np.nan

    def set_data_parameters(self, nrows, minority_class):
        self.nrows = nrows
        self.minority_class = minority_class

    def create_synthetic_data(self):
        
        # Create the synthetic data
        df = pd.DataFrame({'target': [0]*self.nrows})

        # Set target fraction
        index = df.sample(frac=self.minority_class, random_state=0).index
        df.at[index, 'target'] = 1

        # Create the synthetic boolean feature(s)
        for feature_fraction in self.feature_fractions:

            # Boolean feature construction
            column = 'feature_'+str(round(feature_fraction, 3))
            df[column] = 0

            # Set feature fraction
            index = df.sample(frac=feature_fraction, random_state=42).index
            df.at[index, column] = 1
        
        self.data = df
        
    def calculate_null_rigs(self, permutations):
        
        self.permutations = permutations
        
        df_shuffled = self.data.copy()
        
        null_rigs = []

        # Random seed isn't set
        for n in range(self.permutations):
            df_shuffled['target'] = df_shuffled['target'].sample(frac=1).values

            null_rigs.append(rig_all_columns(df_shuffled, 'target'))

        # Get the feature list from the columns
        columns = self.data.columns.tolist()
        columns.remove('target')

        self.null_rigs = pd.DataFrame(null_rigs, columns=columns)
        
        # Get all null RIGs (if there are multiple features)
        values = []
        for column in self.null_rigs.columns:
            values.append(self.null_rigs[column])

        s1 = pd.concat(values)

        self.null_rig_values = s1
        
        n_greater = 1
        quantile = 1-(n_greater/self.permutations)
        print('Max null RIG \t\t= {}'.format(s1.max()))
        print('Median null RIG \t= {}'.format(s1.median()))
        print('{} in {} null RIG \t= {}'.format(n_greater, self.permutations, s1.quantile(quantile)))
        print('Gain threshold \t\t= {}'.format(self.gain_threshold))
        
    def null_rigs_comparison(self):

        counts = (self.null_rig_values > self.gain_threshold).value_counts()

        if True in counts.index:
            print('{} in {} null RIGs greater than {} threshold'.format(counts[True], self.permutations,
                                                                        self.gain_threshold))
        else:
            print("0 in {}".format(self.permutations))
