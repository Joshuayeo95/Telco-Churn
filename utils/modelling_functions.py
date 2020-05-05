# Standard Tools
from scipy import stats
import pandas as pd
import numpy as np
import pickle

# Machine Learning
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC, SVC

# Sampling Tools
from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline

# Visualisation
from yellowbrick.classifier import DiscriminationThreshold
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns



def save_cv_results(cv_results, scoring_name='score', flip_scores=False, verbose=True):
    '''Function to save the training and testing results from cross validation into a dataframe.
    
    Arguments:
        cv_results : dict
            Dictionary of results from scikit-learn's cross_validate method.
        scoring_name : str, default = 'score'
            Name of scorer used in the cross_validate method. If no custom scorer was passed, default should be 'score'.
            In the cv_results dictionary, there should be keys 'train_score' and 'test_score'
            If custom scorer was passed as the scoring method, the cv_results dictionary should have 'train_scoring_name'.
        flip_scores : bool, default = False
            Pass as true if custom scorer is a loss function, which causes scikit learn to return negative scores.
        verbose : bool, default = True
            Prints the mean training and testing scores and fitting times. 
            
    Returns:
        results_df : Pandas DataFrame with training and test scores.
    
    '''
    train_key = 'train_' + scoring_name
    test_key = 'test_' + scoring_name
    
    if flip_scores:
        train_scores = [-result for result in cv_results[train_key]]
        test_scores = [-result for result in cv_results[test_key]]
    else:
        train_scores = cv_results[train_key]
        test_scores = cv_results[test_key]
    
    indices = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    
    results_df = pd.DataFrame({'TrainScores' : train_scores, 'TestScores' : test_scores}, index=indices)
    
    if verbose:
        avg_train_score = np.mean(train_scores)
        avg_test_score = np.mean(test_scores)
        avg_training_time = np.mean(cv_results['fit_time'])
        avg_predict_time = np.mean(cv_results['score_time'])
        
        title = 'Cross Validation Results Summary'
        print(title)
        print('=' * len(title))
        print(f'Avg Training {scoring_name}', '\t', '{:.6f}'.format(avg_train_score))
        print(f'Avg Testing {scoring_name}', '\t', '{:.6f}'.format(avg_test_score))
        print()
        print('Avg Fitting Time', '\t', '{:.4f}s'.format(avg_training_time))
        print('Avg Scoring Time', '\t', '{:.4f}s'.format(avg_predict_time))
    
    return results_df



def training_vs_testing_plot(results, figsize=(5,5), title_fs=18, legend_fs=12):
    '''Function that plots the training and testing scores obtained from cross validation.
    '''
    
    fig = plt.figure(figsize=figsize)
    plt.style.use('fivethirtyeight')
    plt.title('Cross Validation : Training vs Testing Scores', y=1.03, x=0.6, size=title_fs)
    plt.plot(results.TrainScores, color='b', label='Training')
    plt.plot(results.TestScores, color='r', label='Testing')
    plt.legend(loc='center left', bbox_to_anchor=(1.02,0.5), ncol=1, fontsize=legend_fs)
    plt.show();



def cross_validation_summary(model, X_train, y_train, cv_kwargs, flip_scores=False):
    ''' Function that does cross validation and returns a summary report.
        Also plots the trainig and testing scores during cross validation.
        
    Arguments : 
        model : Scikit Learn estimator object
        cv_kwargs : dict
            Cross validation kwargs
        flip_scores : bool, default = False
            Pass as true if custom loss function used for scoring insteatd of existing scikit learn scorers.
    
    Returns:
        cv_results : dict
            Cross validation results from cross_validate method.
    '''
    cv_results = cross_validate(model, X_train, y_train, **cv_kwargs)

    results_df = save_cv_results(cv_results, flip_scores=flip_scores)

    training_vs_testing_plot(results_df)
    
    return cv_results



def get_best_estimator(cv_results, scoring_name='score', greater_is_better=True):
    ''' Function that returns the best estimator found during cross valiation.
    Arguments:
        cv_results : dict
            Results from Sklearn's cross_validate method.
        scoring_name : str, default = 'score'
            Pass name of scorer if custom scorer was used instead of existing scikit learn scorers. 
        greater_is_better : bool, default = True
            Indicates whether higher scores are better. Determines how to select best model.
    
    Returns:
        best_estimator : Sklearn estimator object
            Best estimator found during cross validation.
    '''
    test_key = 'test_' + scoring_name
    scores = list(cv_results[test_key])

    if greater_is_better:
        max_score_index = scores.index(max(scores))
        best_estimator = cv_results['estimator'][max_score_index]

    else:
        min_score_index = scores.index(min(scores))
        best_estimator = cv_results['estimator'][min_score_index]
    
    return best_estimator


def plot_confusion_matrix(y_test, y_pred, figsize=(5,5), ax=None, **heatmap_kwargs):
    '''Function to plot the confusion matrix.
    '''
    fig = plt.figure(figsize=figsize)
    
    if ax == None:
        ax = plt.gca()
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, cmap='Blues', annot=True, annot_kws={'size':16}, fmt='g', ax=ax)
    ax.set_title('Confusion Matrix', size=18, y=1.05)
    ax.set_ylabel('Actual Class', size=16)
    ax.set_xlabel('Predicted Class', size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    
    plt.show();


def model_predictions(model, X_train, y_train, X_test, threshold=0.5, predict_proba=True):
    '''Function that fits a Sklearn estimator object and returns predictions.
    Arguments:
        model : Sklearn estimator object
        predict_proba : bool, default = True
            Returns class probabilities for binary classifiers
    Returns:
        y_pred : array
            Predicted target values
        y_proba : array
            Predicted class probabilities for the positive class
    '''
    model.fit(X_train, y_train)
    if threshold == 0.5:
        y_pred = model.predict(X_test)
    else:
        y_pred = np.where(model.predict_proba(X_test)[:,1] > threshold, 1, 0)
    
    if predict_proba:
        y_proba = model.predict_proba(X_test)[:,1]
        return y_pred, y_proba
    
    else:
        return y_pred


def holdout_evaluation(model, X_train, y_train, X_test, y_test, threshold=0.5, model_name='', ax=None):
    ''' Function to obtain holdout data predictions and plot the confusion matrix.
    
    Arguments:
        threshold : float, default = 0.5
            Threshold for class prediction using probabilies.
        model_name : str
            Name of model to be printed on the console.
        conf_matrix : bool, default = True
            Plots the confusion matrix for the holdout datset.
    Returns:
        y_pred : Pandas Series
            Model predictions for the holdout dataset.
    '''
    y_pred = model_predictions(model, X_train, y_train, X_test, threshold=threshold, predict_proba=False)
    
    report_title = f'Holdout Dataset Classification Report for {model_name}'  
    print(report_title)
    print('=' * len(report_title))
    print(classification_report(y_test, y_pred))
    print()
    
    plot_confusion_matrix(y_test, y_pred, ax=ax)
    
    return y_pred
    

def classification_scores(y_test, y_pred, model_name, decimal_places=4):
    ''' Function that calculates the classification scores of the model and saves them into a dataframe.
    '''
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) 
    
    df = pd.DataFrame({
        'Model' : [model_name],
        'Accuracy' : [accuracy],
        'ROC_AUC' : [roc_auc],
        'Precision' : [precision],
        'Recall' : [recall],
        'F1' : [f1]
    })
    
    df = df.round(decimal_places)
    
    return df


def param_tuning_plots(model, param_grid, X_train, y_train, gs_kwargs, flip_scores=False, n_cols=2, figsize=(5,5), x_axis_log_scale=False, tight_layout=True):
    ''' Function that plots the average training and testing scores during cross validation for
    each parameter in the parameter grid.
    
    Arguments:
        model : scikit learn estimator
        param_grid : dict
            Format is {parameter : [range of values to iterate over]}
        gs_kwargs : dict
            Kwargs for GridSearchCV function (scoring metric / cross validation method / return_estimator). 
        flip_scores : bool, default = False
            Pass as True if scoring is based on a loss function.
        n_cols : int
            Number of columns for the grid plot.
    '''
    n_vars = len(param_grid)
    n_rows = int(np.ceil(n_vars / n_cols))
    index = 0
    
    fig = plt.figure(figsize=figsize)
    plt.suptitle('Training vs Testing Scores', y=1.03, size=16)
    
    for param, param_range in dict.items(param_grid):
        gridsearch = GridSearchCV(model, param_grid={param : param_range}, **gs_kwargs)
        gridsearch.fit(X_train, y_train)
        cv_results = gridsearch.cv_results_
        
        if flip_scores:
            train_scores = [-result for result in cv_results['mean_train_score']]
            test_scores = [-result for result in cv_results['mean_test_score']]
        else:
            train_scores = cv_results['mean_train_score']
            test_scores = cv_results['mean_test_score']    
        
        results_df = pd.DataFrame({'TrainScores' : train_scores, 'TestScores' : test_scores},
                                  index=param_range)
        
        ax = fig.add_subplot(n_rows, n_cols, index+1)
        plt.plot(results_df.TrainScores, color='b', label='Training')
        plt.plot(results_df.TestScores, color='r', label='Testing')

        if x_axis_log_scale:
            plt.xscale('log')

        plt.xlabel(param, size=14)
        plt.legend(loc='center left', bbox_to_anchor=(1.02,0.5), ncol=1, prop={'size': 14})

        index += 1
    
    if tight_layout:
        plt.tight_layout()
    
    plt.show();