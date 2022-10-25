import pandas as pd
import numpy as np
import datetime as dt
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import random
import math
from sklearn.preprocessing import label_binarize
#constants


#models to compare
models = [
          ('LogReg', LogisticRegression()),
          ('RF', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC()),
          ('GNB', GaussianNB()),
          ('XGB', XGBClassifier())
        ]


#get data
filepath =r'C:\NewPycharmProjects\FuturesResearch\DATA\NQ\NQ_2010-2020_60min.csv'


def fill_missing(data):        #fill any missing / NaN / inf values in the data
    data = np.where(data == np.inf, np.nan, data)
    data = np.where(data == -np.inf, np.nan, data)
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=10, weights='distance')
    data = pd.DataFrame(imputer.fit_transform(data))
    # data = np.where(abs(data - np.mean(data).mean()) > 5 * np.std(data).std(), np.nan, data)
    imputer = KNNImputer(missing_values = np.nan, n_neighbors = 10, weights = 'distance')
    filledData = pd.DataFrame(imputer.fit_transform(data))
    if filledData.isnull().sum().sum() > 0:
        raise ValueError('DATA STILL CONTAINS NAN VALUES')
    return filledData


def prepare_data(filepath):
    data = pd.read_csv(filepath)
    data.drop('Time', axis = 1, inplace = True)
    if 'Time.1' in data.columns and math.isnan(data.loc[0, 'Time.1']):
        data.drop('Time.1', axis = 1, inplace = True)
    if 'Open Int' in data.columns and math.isnan(data.loc[0, 'Open Int']):
        data.drop('Open Int', axis=1, inplace=True)
    data['relChange'] = data['Change']/data['Open']
    X = fill_missing(data)
    y = data['relChange'].shift(-1)

    oldy = y[:-1].copy().abs()

    X = X.iloc[:-1, :]
    for i,c in enumerate(y):
        y[i] = 1 if c>0 else -1
    y = y[:-1]

    # XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=.25, random_state=10)
    XTrain, XTest, yTrain, yTest = X.iloc[:round(len(X)*3/4), :] , X.iloc[round(len(X)*3/4):, :], y.iloc[:round(len(X)*3/4)] , y.iloc[round(len(X)*3/4):],
    oldy = oldy[yTest.index].reset_index(drop=True)
    XTest = XTest.reset_index(drop=True)
    yTest = yTest.reset_index(drop=True)
    return XTrain, XTest, yTrain, yTest, oldy
    

    # cols = featureDF.columns.tolist()
    # del cols[cols.index('strength')]
    # featureDF = featureDF.loc[:,cols]
    # X = fill_missing(featureDF)
    # XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=.25, random_state=69)
    # return XTrain, XTest, yTrain, yTest


def run_exps(XTrain: pd.DataFrame, XTest: pd.DataFrame, yTrain: pd.DataFrame, yTest: pd.DataFrame, oldy: pd.Series) -> pd.DataFrame:
    '''
    Lightweight script to test many models and find winners
    :param XTrain: training split
    :param yTrainain: training target vector
    :param XTest: test split
    :param yTest: test target vector
    :return: DataFrame of predictions
    '''

    dfs = []
    results = []
    names = []

    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    target_names = yTrain

    for name, model in models:
        # try:
        #     kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=90210)
        #     cvResults = model_selection.cross_validate(model, XTrain, yTrain, cv=kfold, scoring=scoring)
        clf = model.fit(XTrain, yTrain)
        yPred = clf.predict(XTest)
        print(name)
        profit = 0
        score = 0
        for i, item in enumerate(yPred):
            if  item  == yTest[i]:
                profit += oldy[i]
                score += 1
            else:
                profit -= oldy[i]
                continue

        print(score / len(yPred))
        print(profit)
        time.sleep(2)
            # print(classification_report(yTest, yPred, target_names=target_names))
    #

    results.append(cvResults)
    names.append(name)
    thisDF = pd.DataFrame(cvResults)
    thisDF['model'] = name
    dfs.append(thisDF)
    final = pd.concat(dfs, ignore_index=True)

    return final


def eval(final: pd.DataFrame):
    rng = np.random.default_rng(69)
    bootstraps = []
    for model in list(set(final.model.values)):
        modelDF = final.loc[final.model == model]
        bootstrap = modelDF.sample(n=30, replace=True)
        bootstraps.append(bootstrap)

    bootstrapDF = pd.concat(bootstraps, ignore_index=True)
    resultsLong = pd.melt(bootstrapDF, id_vars=['model'], var_name='metrics', value_name='values')
    timeMetrics = ['fit_time', 'score_time']  # fit time metrics
    ## PERFORMANCE METRICS
    resultsLongNoFit = resultsLong.loc[~resultsLong['metrics'].isin(timeMetrics)]  # get df without fit data
    resultsLongNoFit = resultsLongNoFit.sort_values(by='values')
    ## TIME METRICS
    resultsLongFit = resultsLong.loc[resultsLong['metrics'].isin(timeMetrics)]  # df with fit data
    resultsLongFit = resultsLongFit.sort_values(by='values')

    print(f'the process took {dt.datetime.now() - tic}')
    plt.figure(figsize=(20, 12))
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=resultsLongNoFit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Classification Metric')
    plt.savefig('./benchmark_models_performance.png', dpi=300)
    plt.show()

    plt.figure(figsize=(20, 12))
    sns.set(font_scale=2.5)
    g = sns.boxplot(x="model", y="values", hue="metrics", data=resultsLongFit, palette="Set3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Comparison of Model by Fit and Score Time')
    plt.savefig('./benchmark_models_time.png', dpi=300)
    plt.show()

    metrics = list(set(resultsLongNoFit.metrics.values))
    metricDF = bootstrapDF.groupby(['model'])[metrics].agg([np.std, np.mean])

    timeMetrics = list(set(resultsLongFit.metrics.values))
    timeDF = bootstrapDF.groupby(['model'])[timeMetrics].agg([np.std, np.mean])

    return metricDF, timeDF

tic = dt.datetime.now()
XTrain, XTest, yTrain, yTest, oldy = prepare_data(filepath)

fDF  = run_exps(XTrain, XTest, yTrain, yTest, oldy)
eval(fDF)

