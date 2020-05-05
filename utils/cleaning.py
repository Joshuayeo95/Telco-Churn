import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

## TODO: Functions for cleaning headers, creating new time variables, show variables with missing values, 


def clean_headers(df):
    '''Function that formats the column headers of a dataframe.
    Arguments:
        df : Pandas DataFrame
        
    Returns:
        df : Pandas DataFrame
            Formatted dataframe with headers cleaned.
    
    '''
    print('Cleaning data frame columns ...')

    df.columns = df.columns.str.replace(' ', '')
    print('Removing whitespaces ...')
    
    df.columns = df.columns.str.replace('-', '_')
    print('Replacing dashes with underscores ...')

    df.columns = df.columns.str.lower()
    print('Changing to lower case ...')
    
    print('Dataframe column headers have been formatted.')

    return df


def data_report(df):
    '''Function that performs preprocessing checks of the data and returns what problems need to be addressed.
    
    Arguments:
        df : Pandas DataFrame

    Returns:
        df : Formatted Pandas DataFrame
    '''
    report_title = 'SUMMARY REPORT FOR DATASET'
    print('=' * len(report_title))
    print(report_title)
    print('=' * len(report_title))
    print()

    # Removing whitespace, replacing dashes and lowercase 
    df = clean_headers(df) 
    print()

    # Overall properties of the data frame 
    numerical_features = df.select_dtypes(include='number').columns.to_list()
    categorical_features = df.select_dtypes(exclude='number').columns.to_list()

    print(f'The data has {df.shape[0]} observations and {df.shape[1]} features.')
    print(f'Numerical Features : {len(numerical_features)}')
    print(f'Categorical Features : {len(categorical_features)}')
    print()

    missing_values = df.isnull().values.any()

    if missing_values:
        print('There data has missing values.')
        print('To save the features and their respective missing values, call the variable_missing_percentage function.')
        print()
        print('Methods to handle missing values for Numerical Features:')
        print('1. Impute missing values with Mean / Median / KNN Imputer')
        print('2. Drop feature with high proportion of missing values.')
        print('3. For time series data, consider Linear Interpolation / Back-filling / Forward-filling')
        print()
        print('Methods to handle missing values for Categorical Features:')
        print('1. Impute missing values with Mode / KNN Imputer')
        print('2. Classify all the missing values as a new category.')
        print('3. Drop feature with high proportion of missing values.')
        print()
    else:
        print('There data has no missing values.')
        print()
    
    # Checking skewed numerical variables
    skewness = df[numerical_features].apply(lambda x : np.abs(stats.skew(x)))
    skewed_vars = skewness.loc[skewness >= 1]

    if len(skewed_vars) > 0:
        print('There are Numerical Features within the data that have skewness greater than 1 in magnitude.')
        print('- To save a list of the skewed variables, call the function check_variable_skew.')
        print('- To visualise the distributions of the skewed features, call the function skewness_subplots.')
        print()
    if len(skewed_vars) == 0:
        print('There are no Numerical Features that are skewed.')
        print()

    print('Reminder : Check feature data types are correct.')
    print('Reminder : Check for narrowly distributed variables - most_frequent_value_proportion function')
    print('Reminder : Check the count plots for categorical variables.')
    print('Reminder : Check for potential outliers - Boxplots / Scatterplots')
    print('Reminder : Plot the Correlation Heatmap.')
    print('1. For continuous - continuous correlations, use Pearsons Correlation.')
    print('2. For binary - continuous correlations, use Point Biserial Correlation.')
    print('3. For ordinal - continuous correlations, use Spearmans Rho.')
    print()
    
    report_ending = 'END OF REPORT'
    print ('=' * len(report_ending))
    print(report_ending)
    print ('=' * len(report_ending))

    return df
    

def numerical_categorical_split(df):
    '''Function that creates a list for the numeric variables and categorical variables respectively.
    '''
    numerical_var_list = df.select_dtypes(include='number').columns.to_list()
    categorical_var_list = df.select_dtypes(exclude='number').columns.to_list()
    
    return numerical_var_list, categorical_var_list



def change_variables_to_categorical(df, vars_to_change=[]):
    '''Function that changes all non-numeric variables to categorical datatype.
    
    Arguments:
        df : Pandas DataFrame
        vars_to_change : list, default is an empty list
            If a non-empty list is passed, only the variables in the list are converted to categorical datatype.
    
    Returns:
        df : Pandas DataFrame with categorical datatypes converted.
    '''
    categorical_variables = df.select_dtypes(exclude='number').columns.to_list()
    
    if len(vars_to_change) > 0:
        categorical_variables = vars_to_change
    
    for var in categorical_variables:
        df[var] = df[var].astype('category')
        
    return df
    





def variable_missing_percentage(df, verbose=True, save_results=False):
    '''
    Function that shows variables that have missing values and the percentage of total observations that are missing.
    
    Arguments:
        df : Pandas DataFrame
        verbose : bool, default = True
            Print the variables that have missing data in descending order.
        save_results : bool, default = False
            Set as True to save the missing percentages.
    
    Returns:
        percentage_missing : Pandas Series
            Series with variables and their respective missing percentages.
    '''
    print(f'The dataframe has {df.shape[1]} variables.')

    percentage_missing = df.isnull().mean().sort_values(ascending=False) * 100
    percentage_missing = percentage_missing.loc[percentage_missing > 0].round(2)
    missing_variables = len(percentage_missing)
    
    if verbose:
        if len(percentage_missing) > 0:
            print(f'There are a total of {missing_variables} variables with missing values and their missing percentages are as follows:')
            print()
            print(percentage_missing)
        
        else:
            print('The dataframe has no missing values in any column.')
    
    if save_results:
        return percentage_missing



def check_negative(df, exclude_vars=[]):
    '''Function that check the numeric columns and determines if they are all non-negative.
    Arguments:
        df : Pandas DataFrame
            Dataframe to be checked.
        exclude_vars : list
            List of variables names to exclude from the check.
    '''
    numeric_vars = df.select_dtypes(include='number').columns.to_list()
    vars_to_check = [var for var in numeric_vars if var not in exclude_vars]
    neg_values = [] 

    for var in vars_to_check:
        neg_values.append(any(df[var] < 0))

    results = pd.DataFrame.from_dict(
        dict(zip(vars_to_check, neg_values)),
        orient='index',
        columns=['NegativeValues']
    )

    if results.NegativeValues.mean() > 0:
        neg_vars_count = results.NegativeValues.loc[results.NegativeValues == 1].count()
        print(f'There are {neg_vars_count} variables with negative values.')
        print('To view variables, please save results.')
    
    else:
        print('There are no numeric variables with negative values.')
    
    return results
    

def drop_missing_variables(df, threshold, verbose=True):
    '''Function that removes variables that have missing percentages above a threshold.
    
    Arguments:
        df : Pandas DataFrame
        threshold : float
            Threshold missing percentage value in decimals.
        verbose : bool, default is True
            Prints the variables that were removed.
            
    Returns:
        df : Pandas DataFrame with variables removed
    '''
    shape_prior = df.shape
    vars_to_remove = df.columns[df.isnull().mean() > threshold].to_list()
    df = df.drop(vars_to_remove, axis=1)
    shape_post = df.shape
    
    print(f'The original DataFrame had {shape_prior[1]} variables.')
    print(f'The returned DataFrame has {shape_post[1]} variables.')
    
    if verbose:
        print()
        print('The following variables were removed:')
        print(vars_to_remove)
        
    return df


def create_time_vars(df, time_var, year=True, month=True, day=True, season=True, drop=True):
    '''Function that creates additional time-related features by extracting them from a datetime series (colunn).
    Arguments:
        df : Pandas DataFrame
            Dataframe with the datetime variable.
        time_var : str
            Variable name. Variable must be in datetime format.
        year : bool
            Creates a new year column in the dataframe.
        month : bool
            Creates a new month column in the dataframe.
        day : bool
            Creates a new day of week column in the dataframe.
        season : bool
            Creates a new season column in the dataframe.
        drop : bool
            Drop time_var which was used to extract the other features.
    
    Returns:
        df : Pandas DataFrame
            Dataframe with added time variables.

    '''
    ## TODO : Raise exceptions for wrong data types.

    if year:
        df['year'] = df[time_var].dt.year
        df.year = df.year.astype('category')

    if month:
        df['month'] = df[time_var].dt.month_name()
        df.month = df.month.astype('category')

    if day:
        df['day'] = df[time_var].dt.day_name()
        df.day = df.day.astype('category')

    if season:
        seasons = [
            'Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer',
            'Summer', 'Summer', 'Autumn', 'Autumn', 'Autumn', 'Winter'
        ]

        month_to_season = dict(zip(range(1,13), seasons))
        df['season'] = df[time_var].dt.month.map(month_to_season)
        df.season = df.season.astype('category')

    if drop:
       df = df.drop(time_var, axis=1)
       print('Datetime variable has been dropped.')
    
    return df


def convert_to_categorical(df, add_vars=[]):
    '''Function to change variable datatypes to categorical data type.
    Arguments:
        df : Pandas DataFrame
            Dataframe to format.
        add_vars : list of variable names 
            Additional variables to change their data type to categorical.
        
    Returns:
        df : Pandas DataFrame
            Formated Dataframe.

    '''
    cat_vars = df.select_dtypes(exclude='number').columns.to_list()
    
    if len(add_vars) > 0:
        [cat_vars.append(x) for x in add_vars]
    
    for var in cat_vars:
        df[var] = df[var].astype('category')
    
    return df


def label_encode(df,exclude_list=[]):
    ''' Funciton that label encodes categorical varaibles that have two or less unique categories.
    Arguments:
        df : Pandas DataFrame
            Dataframe of which its variables will be encoded.
        exclude_list : list
            List of variable names to be excluded from the encoding.
    
    Returns:
        df : Pandas DataFrame
            Dataframe with variables encoded.
    ''' 
    cat_vars = df.select_dtypes(include='category').columns.to_list()
    vars_to_encode = [x for x in cat_vars if x not in exclude_list]
    encoder = LabelEncoder()

    for var in vars_to_encode:
        if len(list(df[var].unique())) <= 2:
            df[var] = encoder.fit_transform(df[var])

    print(f'A total of {len(vars_to_encode)} variables have been encoded.')

    return df

    
    