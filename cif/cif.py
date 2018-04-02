# COMPOSITE INDICATORS
# FUNCTIONS

import os
import requests as rq
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.tsa.x13 as smX13
import statsmodels.tsa.arima_model as smARIMA
#import statsmodels.tsa.statespace.sarimax as smSARIMAX
import statsmodels.tsa.filters.hp_filter as smHP
from dateutil.relativedelta import relativedelta
#from subprocess import call
#from pathlib import Path
#import numbers
import warnings


# OECD API FUNCTIONS

def makeOECDRequest(dsname, dimensions, params = None, root_dir = 'http://stats.oecd.org/SDMX-JSON/data'):
    
    """
    Make URL for the OECD API and return a response.
    
    Parameters
    -----
    dsname: str
        dataset identifier (e.g., MEI for main economic indicators)
    dimensions: list
        list of 4 dimensions (usually location, subject, measure, frequency)
    params: dict or None
        (optional) dictionary of additional parameters (e.g., startTime)
    root_dir: str
        default OECD API (https://data.oecd.org/api/sdmx-json-documentation/#d.en.330346)
        
    Returns
    -----
    results: requests.Response
        `Response <Response>` object
    
    """
    
    if not params:
        params = {}
    
    dim_args = ['+'.join(d) for d in dimensions]
    dim_str = '.'.join(dim_args)
    
    url = root_dir + '/' + dsname + '/' + dim_str + '/all'
    
    print('Requesting URL ' + url)
    return rq.get(url = url, params = params)

    
def getOECDJSONStructure(dsname, root_dir = 'http://stats.oecd.org/SDMX-JSON/dataflow', showValues = [], returnValues = False):
    
    """
    Check structure of OECD dataset.
    
    Parameters
    -----
    dsname: str
        dataset identifier (e.g., MEI for main economic indicators)
    root_dir: str
        default OECD API structure uri
    showValues: list
        shows available values of specified variable, accepts list of integers
        which mark position of variable of interest (e.g. 0 for LOCATION)
    returnValues: bool
        if True, the observations are returned
        
    Returns
    -----
    results: list
        list of dictionaries with observations parsed from JSON object, if returnValues = True
        
    """ 
    
    url = root_dir + '/' + dsname + '/all'
    
    print('Requesting URL ' + url)
    
    response = rq.get(url = url)
    
    if (response.status_code == 200):
        
        responseJson = response.json()
        
        keyList = [item['id'] for item in responseJson.get('structure').get('dimensions').get('observation')]
        
        print('\nStructure: ' + ', '.join(keyList))
        
        for i in showValues:
            
            print('\n%s values:' % (keyList[i]))
            print('\n'.join([str(j) for j in responseJson.get('structure').get('dimensions').get('observation')[i].get('values')]))
            
        if returnValues:
        
            return(responseJson.get('structure').get('dimensions').get('observation'))
        
    else:
        
        print('\nError: %s' % response.status_code)
        

def createOneCountryDataFrameFromOECD(country = 'CZE', dsname = 'MEI', subject = [], measure = [], frequency = 'M', startDate = None, endDate = None):      
    
    """
    Request data from OECD API and return pandas DataFrame. This works with OECD datasets
    where the first dimension is location (check the structure with getOECDJSONStructure()
    function).
    
    Parameters
    -----
    country: str
        country code (max 1, use createDataFrameFromOECD() function to download data from more countries),
        list of OECD codes available at http://www.oecd-ilibrary.org/economics/oecd-style-guide/country-names-codes-and-currencies_9789264243439-8-en
    dsname: str
        dataset identifier (default MEI for main economic indicators)
    subject: list
        list of subjects, empty list for all
    measure: list
        list of measures, empty list for all
    frequency: str
        'M' for monthly and 'Q' for quaterly time series
    startDate: str of None
        date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations
    endDate: str or None
        date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations
        
    Returns
    -----
    data: pandas.DataFrame
        data downloaded from OECD
    subjects: pandas.DataFrame
        subject codes and full names
    measures: pandas.DataFrame
        measure codes and full names
        
    """
    
    # Data download
    
    response = makeOECDRequest(dsname
                                 , [[country], subject, measure, [frequency]]
                                 , {'startTime': startDate, 'endTime': endDate, 'dimensionAtObservation': 'AllDimensions'})
    
    # Data transformation
    
    if (response.status_code == 200):
        
        responseJson = response.json()
        
        obsList = responseJson.get('dataSets')[0].get('observations')
        
        if (len(obsList) > 0):
            
            if (len(obsList) >= 999999):
                print('Warning: You are near response limit (1 000 000 observations).')
        
            print('Data downloaded from %s' % response.url)
            
            timeList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'TIME_PERIOD'][0]['values']
            #subjectList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'SUBJECT'][0]['values']
            #measureList = [item for item in responseJson.get('structure').get('dimensions').get('observation') if item['id'] == 'MEASURE'][0]['values']
            subjectList = responseJson.get('structure').get('dimensions').get('observation')[1]['values']
            measureList = responseJson.get('structure').get('dimensions').get('observation')[2]['values']
            
            obs = pd.DataFrame(obsList).transpose()
            obs.rename(columns = {0: 'series'}, inplace = True)
            obs['id'] = obs.index
            obs = obs[['id', 'series']]
            obs['dimensions'] = obs.apply(lambda x: re.findall('\d+', x['id']), axis = 1)
            obs['subject'] = obs.apply(lambda x: subjectList[int(x['dimensions'][1])]['id'], axis = 1)
            obs['measure'] = obs.apply(lambda x: measureList[int(x['dimensions'][2])]['id'], axis = 1)
            obs['time'] = obs.apply(lambda x: timeList[int(x['dimensions'][4])]['id'], axis = 1)
            #obs['names'] = obs['subject'] + '_' + obs['measure']
            
            #data = obs.pivot_table(index = 'time', columns = ['names'], values = 'series')
            
            data = obs.pivot_table(index = 'time', columns = ['subject', 'measure'], values = 'series')
            
            return(data, pd.DataFrame(subjectList), pd.DataFrame(measureList))
        
        else:
        
            print('Error: No available records, please change parameters')
            return(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    else:
        
        print('Error: %s' % response.status_code)
        return(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def createDataFrameFromOECD(countries = ['CZE', 'AUT', 'DEU', 'POL', 'SVK'], dsname = 'MEI', subject = [], measure = [], frequency = 'M', startDate = None, endDate = None):
    
    """
    Request data from OECD API and return pandas DataFrame. This works with OECD datasets
    where the first dimension is location (check the structure with getOECDJSONStructure()
    function).
    
    Parameters
    -----
    countries: list
        list of country codes, list of OECD codes available at http://www.oecd-ilibrary.org/economics/oecd-style-guide/country-names-codes-and-currencies_9789264243439-8-en
    dsname: str
        dataset identifier (default MEI for main economic indicators)
    subject: list
        list of subjects, empty list for all
    measure: list
        list of measures, empty list for all
    frequency: str
        'M' for monthly and 'Q' for quaterly time series
    startDate: str or None
        date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations
    endDate: str or None
        date in YYYY-MM (2000-01) or YYYY-QQ (2000-Q1) format, None for all observations
        
    Returns
    -----
    data: pandas.DataFrame
        data downloaded from OECD
    subjects: pandas.DataFrame
        subject codes and full names
    measures: pandas.DataFrame
        measure codes and full names
        
    """        
    
    dataAll = pd.DataFrame()
    subjectsAll = pd.DataFrame()
    measuresAll = pd.DataFrame()
    
    for country in countries:
        
        dataPart, subjectsPart, measuresPart = createOneCountryDataFrameFromOECD(country = country, dsname = dsname, subject = subject, measure = measure, frequency = frequency, startDate = startDate, endDate = endDate)
        
        if (len(dataPart) > 0):
            
            dataPart.columns = pd.MultiIndex(levels = [[country], dataPart.columns.levels[0], dataPart.columns.levels[1]],
                labels = [np.repeat(0, dataPart.shape[1]), dataPart.columns.labels[0], dataPart.columns.labels[1]], 
                names = ['country', dataPart.columns.names[0], dataPart.columns.names[1]])
            
            dataAll = pd.concat([dataAll, dataPart], axis = 1)
            subjectsAll = pd.concat([subjectsAll, subjectsPart], axis = 0)
            measuresAll = pd.concat([measuresAll, measuresPart], axis = 0)
    
    if (len(subjectsAll) > 0):
        
        subjectsAll.drop_duplicates(inplace = True)
        
    if (len(measuresAll) > 0):
        
        measuresAll.drop_duplicates(inplace = True)
    
    return(dataAll, subjectsAll, measuresAll)


def getOnlyBestMeasure(df, priorityList, countryColName = 'country', subjectColName = 'subject', measureColName = 'measure'):
    
    """
    Select only one measure per subject.
    
    Parameters
    -----
    df: pandas.DataFrame
        output from create_DataFrame_from_OECD()
    priorityList: list
        list of measures sorted by priority
    countryColName: str
        name of country level of pandas multiindex
    subjectColName: str
        name of subject level of pandas multiindex
    measureColName: str
        name of measure level of pandas multiindex
        
    Returns
    -----
    data: pandas.DataFrame
        dataframe with selected columns
        
    """
        
    data = pd.DataFrame()
    subjectMultiInd = df.columns.names.index(subjectColName)
    measureMultiInd = df.columns.names.index(measureColName)
    
    try:
        
        countryMultiInd = df.columns.names.index(countryColName)
        countryList = df.columns.levels[countryMultiInd]
        print("Data with country multiindex level.")
        
    except:
        
        countryList = ['oneCountryOnly']  
        print("Data without country multiindex level.")
    
    for c in countryList:  
        
        if (c != 'oneCountryOnly'):
            
            dfCountryPart = df.select(lambda x: x[countryMultiInd] == c, axis = 1).copy()
            
        else:
            
            dfCountryPart = df.copy()
        
        for i in list(dfCountryPart.columns.get_level_values(subjectMultiInd).unique()):
            
            dfPart = dfCountryPart.select(lambda x: x[subjectMultiInd] == i, axis = 1).copy()
            
            if dfPart.shape[1] > 1: # Several measures of one subject
                
                col = list(dfPart.columns.get_level_values(measureMultiInd).unique()) # returns existing levels only!
                
                ind = True
                j = 0
                while ind and (j < len(priorityList)):
                    
                    selMeasure = priorityList[j]
                    
                    if selMeasure in col:
                        
                        newCol = dfPart.select(lambda x: x[measureMultiInd] == selMeasure, axis = 1)
                        #newCol.columns = ['_'.join(col).strip() for col in newCol.columns.values]
                        data =  pd.concat([data, newCol], axis = 1)
                        
                        ind = False
                    
                    j += 1
                
                if ind:
                    print('Warning: variable %s not selected.' % (i))
            
            elif dfPart.shape[1] == 1: # One measure only
            
                #dfPart.columns = ['_'.join(col).strip() for col in dfPart.columns.values]
                data =  pd.concat([data, dfPart], axis = 1)
        
    return(data)


# TRANSFORMING DATA

def getRidOfMultiindex(df):
    
    """
    Rename the series from multiindex to index.
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame with multiindex
        
    Returns
    -----
    data: pandas.DataFrame
        dataframe with simple index
        
    """
    
    data = df.copy()
    
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    
    return(data)


'''
def useIndexAsColumn(df):
    
    """
    Get DataFrame index as column, so it can be used with ggplot, but keep the original DataFrame intact.
    
    Parameters
    -----
    df:
        pandas DataFrame
    """
    
    df = df.copy()
    df['index'] = df.index
    df.reset_index(drop = True, inplace = True)
    
    return(df)
'''


def renameQuarters(x):
    
    """
    Rename quarters from YYYY-QQ to YYYY-MM format.
    """
        
    if not isinstance(x, str):
        raise ValueError('Parameter x should be a string type.')
    
    if (re.search('Q1', x)):
        x = re.sub('Q1', '02', x)
        
    elif (re.search('Q2', x)):
        x = re.sub('Q2', '05', x)
        
    elif (re.search('Q3', x)):
        x = re.sub('Q3', '08', x)
        
    elif (re.search('Q4', x)):
        x = re.sub('Q4', '11', x)
        
    else:
        print('Warning: Unknown format of index "%s"' % (str(x)))
        
    return(x)


def renameQuarterlyIndex(df):
    
    """
    Change index of pandas DataFrame with quaterly time series, so it matches monthly DataFrames.
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame (quaterly time series)
        
    Returns
    -----
    data: pandas.DataFrame
        dataframe with renamed index
    """
    
    data = df.copy()
    
    ind = data.index
    newInd = [renameQuarters(x) for x in ind]
    
    data.index = newInd
    
    return(data)


def createMonthlySeries(df, divide = True):
    
    """
    Take quarterly time series from pandas DataFrame and convert their
    frequency to months (linear interpolation, aligning with the middle
    month). Return pandas DataFrame with the same number of columns
    as original DataFrame and index in date format.
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame (quaterly series with index in format YYYY-MM-DD)
    divide: bool
        should the series be devided by 3 during interpolation? This should be
        set to true for interval time series, e.g., GDP.
        
    Returns
    -----
    data: pandas.DataFrame
        dataframe with monthly series
        
    """
    
    data = df.copy()
    
    minMonth = int(min(data.index).month)
    minYear = int(min(data.index).year)
    maxMonth = int(max(data.index).month)
    maxYear = int(max(data.index).year)
        
    periods = (maxYear - minYear) * 12 + (maxMonth - minMonth)
    newInd = pd.date_range(start = min(data.index), periods = periods + 1, freq = 'MS')
    
    newData = pd.DataFrame(index = newInd)
    
    for col in data.columns:
        
        ts = data.loc[ : , col]
        
        if divide:
            
            newCol = pd.DataFrame(data = ts/3, index = newInd)
            
        else:
            
            newCol = pd.DataFrame(data = ts, index = newInd)
        
        firstValIndex = newCol.first_valid_index()
        lastValIndex = newCol.last_valid_index()
        
        newCol = newCol[firstValIndex:lastValIndex].interpolate()
        
        newData = pd.concat([newData, newCol], axis = 1)
    
    return(newData)
    

def getIndexAsDate(df):
    
    """
    Take string date index in format YYYY-MM and transform it to date in format YYYY-MM-DD.
    
    Parameters
    -----
    df: pandas.DataFrame
        monthly pandas DataFrame (series with index in format YYYY-MM)
        
    Returns
    -----
    data: pandas.DataFrame
        dataframe with index as date
        
    """
    
    df = df.set_index(pd.to_datetime(df.index, format = '%Y-%m'))
    
    return(df)


def getSAForecasts(series, forecastSteps = 6, showPlots = True, savePlots = None, saveLogs = None):
    
    """
    Get seasonally adjusted time series with forecasts.
    
    Parameters
    -----
    series: pd.DataFrame
        monthly pandas DataFrame (series with index in format YYYY-MM) with one column
    forecastSteps: int
        length of forecast period (default = 6)
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot (only the final plot of forecasted series)
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
        
    Returns
    -----
    data: pandas.DataFrame
        dataframe with seasonally adjusted data and stabilising forecasts
        
    """
    
    try:
        
        # Get best model info
        
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")
            
            series_ord = smX13.x13_arima_select_order(series) 
            print('\nBEST MODEL BY TRAMO:', series_ord.order, series_ord.sorder)
        
        # Get seasonally adjusted data
        
        maxOrder = min(max(series_ord.order[0], series_ord.order[2]), 4) # max from AR and MA orders, no larger than 4
        maxSOrder = min(max(series_ord.sorder[0], series_ord.sorder[2]), 2) # max from seasonal AR and MA orders, no larger than 2
        diff = min(series_ord.order[1], 2) # order of differencing, no larger than 2
        diffS = min(series_ord.sorder[1], 1) # order of seasonal differencing, no larger than 1
        
        with warnings.catch_warnings():
            
            warnings.simplefilter("ignore")
            
            series_X13 = smX13.x13_arima_analysis(endog = series, maxorder = (maxOrder, maxSOrder), maxdiff = None, diff = (diff, diffS), outlier = True, forecast_years = 0)
        
        series_SA = pd.DataFrame({series.columns[0]: series_X13.seasadj})        
        
        # Create short-term forecasts
        
        series_SA_ARIMA_model = smARIMA.ARIMA(series_SA, order = series_ord.order)
        
        series_SA_ARIMA = series_SA_ARIMA_model.fit(disp = 0)
        #print('\nARIMA model with TRAMO parameters:')
        #print(series_SA_ARIMA.summary())
        
        series_SA_forecast = series_SA_ARIMA.forecast(steps = forecastSteps)[0] # use ARIMA with X13 specifications to create forecasts (forecasts with TRAMO-SEATS directly in X13 doesn't work)
        series_SA_forecast = pd.DataFrame(series_SA_forecast
                                      , columns = [series.columns[0]]
                                      , index = pd.date_range(series_SA.index[-1], periods = forecastSteps + 1, freq='MS')[1:]
                                      )
        
        series_SA_withForecast = series_SA.append(series_SA_forecast)
        #series_SA_withForecast.set_index(pd.date_range(series_SA.index[0], periods = len(series_SA) + forecastSteps, freq='MS'), inplace = True)
        
        if showPlots:
            
            series_X13.plot()
            series_SA.plot()
            series_SA_withForecast.plot()
            
        if savePlots:
            # saving only SA with forecast
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 2))
            ax.plot(series_SA_withForecast, color = 'gray')
            fig.savefig(os.path.join(savePlots, str(series.columns[0]) + '_01_SA.png'), dpi = 300)
            plt.close(fig)
            
        if saveLogs:
            
            saveLogs.write('\nBEST MODEL BY TRAMO: ' + str(series_ord.order) + str(series_ord.sorder))
            
    #        warnings = re.split('WARNING: ', str(series_X13.stdout)) ## attention, we now use warnings package!
    #        
    #        if len(warnings) > 1:
    #        
    #            warnings = [warn.replace('\\r\\n', ' ').strip() for warn in warnings[1:]]
    #            saveLogs.write('\nX13 WARNINGS: ' + str(warnings))
            
            saveLogs.write('\nX13 WARNINGS: ' + str(series_X13.stdout).replace('\\r\\n', ' ').strip())
            saveLogs.flush()
        
    except Exception as e:
        
        series_SA_withForecast = series
        
        print('\nBEST MODEL BY TRAMO: None')
        print('\nX13 ERRORS: ' + str(e))
        print('\nWARNING: Returning original time series without forecasts.')
        
        if saveLogs:
            
            saveLogs.write('\nBEST MODEL BY TRAMO: None')
            saveLogs.write('\nX13 ERRORS: ' + str(e).replace('\\r\\n', ' ').strip())        
            saveLogs.write('\nWARNING: Returning original time series without forecasts.')
            saveLogs.flush()
            
        if savePlots:
            # saving only SA with forecast
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 2))
            ax.plot(series_SA_withForecast, color = 'gray')
            fig.savefig(os.path.join(savePlots, str(series.columns[0]) + '_01_SA.png'), dpi = 300)
            plt.close(fig)
    
    return(series_SA_withForecast)
        

def applyHPTwice(series, dateMax = None, lambda1 = 133107.94, lambda2 = 13.93, showPlots = True, savePlots = None, saveAllPlots = False, returnTrend = False):
    
    """
    Apply Hodrick-Prescott filter twice: first to remove the trend,
    second to get rid of seasonality and irregularities.
    
    Parameters
    -----
    series: pandas.DataFrame
        monthly pandas DataFrame (series with index in format YYYY-MM)
        with one column, should be seasonally adjusted
    dateMax: datetime.date or None
        (optional) last date of original dataset, useful when input series
        contains forecasts 
    lambda1: float
        lambda value, that allows to remove components that have a cycle
        length longer than 120 months (default = 133107.94)
    lambda2: float
        lambda value, that allows to remove components that have a cycle
        length shorter than 12 months (default = 13.93)
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot (only the final plot of cycle)
    saveAllPlots: bool
        should all the plots be saved, or just the final series (default)?
        This works only when the path in savePlots is set.
    returnTrend: bool
        if True return trend and cycle, otherwise (default) return only cycle
        
    Returns
    -----
    trend: pandas.DataFrame
        returned, if returnTrend = True, dataframe with trend component
    cycle: pandas.DataFrame
        dataframe with cyclical component
    """
    
    if not(dateMax):
        
        dateMax = series.index[-1]

    series_HP1 = smHP.hpfilter(series, lamb = lambda1)
    series_HP2 = smHP.hpfilter(series_HP1[0], lamb = lambda2)
    series_HP = series_HP2[1][series_HP2[1].index <= dateMax] # without forecasted values
    
    if showPlots:
        
        plotHP(series_HP1)
        plotHP(series_HP2, phase = 2)
        
    if savePlots:
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 2))
        ax.plot(series_HP, color = 'gray')
        fig.savefig(os.path.join(savePlots, str(series_HP.columns[0]) + '_02_HP.png'), dpi = 300)
        plt.close(fig)
        
        if saveAllPlots:
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1)
            plotHP(series_HP1)
            fig.savefig(os.path.join(savePlots, str(series_HP.columns[0]) + '_02a_HP.png'), dpi = 300)
            plt.close(fig)
            
            fig, ax = plt.subplots(nrows = 1, ncols = 1)
            plotHP(series_HP2, phase = 2)
            fig.savefig(os.path.join(savePlots, str(series_HP.columns[0]) + '_02b_HP.png'), dpi = 300)
            plt.close(fig)
    
    if returnTrend:
        return(series_HP1[1], series_HP)
    else:
        return(series_HP)
    

def normaliseSeries(series, createInverse = False, showPlots = True, savePlots = None):

    """
    Normalise and rescale series. Optionally create inverted time series
    to analyze countercyclical series.
    
    Parameters
    -----
    series: pandas.DataFrame
        monthly pandas DataFrame (series with index in format YYYY-MM) with one column
    createInverse: bool
        create inverse time series?
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plots
        
    Returns
    -----
    data: pandas.DataFrame
        dataframe with normalised values
        
    """
    
    mean = series.mean()[0]
    mad = (series - mean).abs().sum()[0] / series.shape[0] # mean absolute deviation
    
    series_norm = ((series - mean) / mad) + 100
    
    if showPlots:
        
        series_norm.plot()
        
    if savePlots:
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 2))
        ax.plot(series_norm, color = 'gray')
        fig.savefig(os.path.join(savePlots, str(series_norm.columns[0]) + '_03_norm.png'), dpi = 300)
        plt.close(fig)
    
    if createInverse:
        
        colName = series.columns[0]
        series_inv_norm = ((series - mean) / mad) * (-1) + 100
        series_inv_norm = series_inv_norm.rename(columns = {colName: str(colName) + '_INV'})
        
        return(series_norm, series_inv_norm)
        
    else:
        
        return(series_norm)


def pipelineOneColumnTransformations(col, showPlots = True, savePlots = None, saveLogs = None, createInverse = False):
    
    """
    Pipeline connecting transformation functions (forecasting, HP filter and
    normalising the series).
    
    Parameters
    -----
    col: pandas.DataFrame
        monthly pandas DataFrame (series with index in format YYYY-MM) with one column
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plots
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
    createInverse: bool
        create inverse time series?
        
    Returns
    -----
    col_SA_withForecast: pandas.DataFrame
        dataframe with seasonally adjusted data with stabilising forecasts
    col_SA_trend: pandas.DataFrame
        dataframe with trend component
    col_SA_HP: pandas.DataFrame
        dataframe with cyclical component
    col_SA_HP_norm: pandas.DataFrame
        dataframe with normalised values of cyclical component
    col_inv_SA_HP_norm: pandas.DataFrame
        dataframe with normalised values of cyclical component of inverted series,
        returned, if createInverse = True
    """
    
    # a) Save plot of the original series
            
    if savePlots:
    
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 2))
        ax.plot(col, color = 'gray')
        fig.savefig(os.path.join(savePlots, str(col.columns[0]) + '_00_orig.png'), dpi = 300)
        plt.close(fig)
    
    
    # b) Seasonal adjustment, outlier filtering and short-term prediction
    
    col_SA_withForecast = getSAForecasts(col, showPlots = showPlots, savePlots = savePlots, saveLogs = saveLogs)
    
    
    # c) Cycle identification (Hodrick-Prescott filter)
    
    if (col_SA_withForecast.shape[0] > 12):
        
        col_SA_trend, col_SA_HP = applyHPTwice(col_SA_withForecast, dateMax = col.index[-1], showPlots = showPlots, savePlots = savePlots, returnTrend = True)
    
    else:
        
        print('Warning: Series shorter than 12 months. No cycle identification possible. No plots created.')
        col_SA_trend = col_SA_withForecast
        col_SA_HP = col_SA_withForecast
        
        showPlots = False # otherwise error in matplotlib
        savePlots = False # otherwise error in matplotlib
        
    
    # d) Normalisation
    
    if createInverse:
        
        col_SA_HP_norm, col_inv_SA_HP_norm = normaliseSeries(col_SA_HP, createInverse = createInverse, showPlots = showPlots, savePlots = savePlots)
        
        return(col_SA_withForecast, col_SA_trend, col_SA_HP, col_SA_HP_norm, col_inv_SA_HP_norm)
    
    else:
        
        col_SA_HP_norm = normaliseSeries(col_SA_HP, createInverse = createInverse, showPlots = showPlots, savePlots = savePlots)
        
        return(col_SA_withForecast, col_SA_trend, col_SA_HP, col_SA_HP_norm)


def pipelineTransformations(df, showPlots = True, savePlots = None, saveLogs = None, createInverse = False):
    
    """
    Pipeline connecting transformation functions (forecasting, HP filter and
    normalising the series) for multiple column data frames.
    If createInverse option is True, then 2 data frames are returned: the first
    one contains all the series (original and inverted), the second one contains
    original series only, which is useful to shorten time needed for turning
    points detection.
    
    Parameters
    -----
    df: pandas.DataFrame
        monthly pandas DataFrame (series with index in format YYYY-MM)
    showPlots: bool
        show plots?
    savePlots: str of None
        path where to save plots
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
    createInverse: bool
        create inverse time series?
    
    Returns
    -----
    data: pandas.DataFrame
        dataframe with normalised values of cyclical components
    
    """
    
    SA_HP_norm = pd.DataFrame(index = df.index)
    
    nCol = df.shape[1]
    
    for i, item in enumerate(df.columns):
        # i, item = list(enumerate(df.columns))[44]
        
        print("\nANALYSING SERIES %d from %d: %s" % (i + 1, nCol, item))
        
        if saveLogs:
            
            saveLogs.write("\n\nANALYSING SERIES %d from %d: %s" % (i + 1, nCol, item))
            saveLogs.flush()
        
        col = pd.DataFrame(df[item][df[item].notnull()], columns = [item])
        
        # Seasonally adjust, detrend and nomalize
        # and add to the DataFrame
        
        if createInverse:
            
            col_SA_withForecast, col_SA_trend, col_SA_HP, col_SA_HP_norm, col_inv_SA_HP_norm = pipelineOneColumnTransformations(col, showPlots = showPlots, savePlots = savePlots, saveLogs = saveLogs, createInverse = createInverse)
            SA_HP_norm = pd.concat([SA_HP_norm, col_SA_HP_norm, col_inv_SA_HP_norm], axis = 1)
            
        else:
            
            col_SA_withForecast, col_SA_trend, col_SA_HP, col_SA_HP_norm = pipelineOneColumnTransformations(col, showPlots = showPlots, savePlots = savePlots, saveLogs = saveLogs, createInverse = createInverse)
            SA_HP_norm = pd.concat([SA_HP_norm, col_SA_HP_norm], axis = 1)
    
    # Return results
        
    return(SA_HP_norm)

    
# BRY-BOSCHAN ALGORITHM

def getLocalExtremes(df, showPlots = True, savePlots = None, nameSuffix = ''):
    
    """
    Find local maxima/minima in df. Mark all point which are higher/lower than their 5 nearest neighbours.
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame (with one column)
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot
    nameSuffix: str
        plot name suffix used when savePlots != None
        
    Returns
    -----
    indicator: pandas.DataFrame
        dataframe with local extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
        
    """
    
    dataShifted = pd.DataFrame(index = df.index)
    
    for i in range(-5, 5):
        
        dataShifted = pd.concat([dataShifted, df.shift(i).rename(columns = {df.columns[0]: 'shift_' + str(i)})], axis = 1)
        
    dataInd = pd.DataFrame(0, index = df.index, columns = df.columns)
    dataInd[dataShifted['shift_0'] >= dataShifted.drop('shift_0', axis = 1).max(axis = 1)] = 1
    dataInd[dataShifted['shift_0'] <= dataShifted.drop('shift_0', axis = 1).min(axis = 1)] = -1
    
    # No extremes near the beginning/end of the series
    
    dataInd[:5] = 0
    dataInd[-5:] = 0
    
    if showPlots or savePlots:
        
        plotIndicator(df, dataInd, showPlots = showPlots, savePlots = savePlots, nameSuffix = nameSuffix)
    
    return(dataInd)


def checkAlterations(df, indicator, keepFirst = False, printDetails = True, showPlots = True, savePlots = None, nameSuffix = '', saveLogs = None):
    
    """
    Check the alterations of the turning points, otherwise delete repeating turning
    points and keep only the first one (if keepFirst = True) or the highest max
    or lowest min (if keepFirst = False, default).
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame (with one column), vector of values
    indicator: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes
    keepFirst: bool
        the first peak or trough is kept if True, the highest peak or the lowest
        trough is kept if False (deault)
    printDetails: bool
        print details about deleted extremes?
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot
    nameSuffix: str
        plot name suffix used when savePlots != None
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
        
    Returns
    -----
    indicator: pandas.DataFrame
        dataframe with extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
        
    """
    
    dataInd = indicator.copy()
    checkAlt = dataInd.cumsum()
    
    if printDetails:
        
        print('\nChecking extremes at %s for alterations:' % (dataInd.columns[0]))
    
    if saveLogs:
        
        saveLogs.write('\nChecking extremes at %s for alterations:' % (dataInd.columns[0]))
    
    if ((checkAlt.max() - checkAlt.min())[0] > 1): # are there any non alterating turning points?
        
        lastExt = 0
        lastDate = df.index[0]
        
        thisDate = dataInd[dataInd != 0].first_valid_index()
        
        while thisDate:
            
            thisExt = dataInd.loc[thisDate][0]
            
            if thisExt == lastExt: # both local extremes of the same type?
            
                if (not(keepFirst) and ((thisExt * df.loc[thisDate])[0] > (lastExt * df.loc[lastDate])[0])): # keep the higher one (or the earlier one when they equal)
                    
                    if printDetails:
                        
                        print('Deleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                    
                    if saveLogs:
                        
                        saveLogs.write('\nDeleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                    
                    dataInd.loc[lastDate] = 0
                    lastExt = thisExt
                    lastDate = thisDate
                    
                else:
                    
                    if printDetails:
                    
                        print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                    
                    if saveLogs:
                        
                        saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                    
                    dataInd.loc[thisDate] = 0
                    
            else: 
                
                lastExt = thisExt
                lastDate = thisDate
                
            try:
                
                thisDate = dataInd[thisDate:][1:][dataInd != 0].first_valid_index()
                
            except IndexError:
                
                break
    
    if showPlots or savePlots:
        
        plotIndicator(df, dataInd, showPlots = showPlots, savePlots = savePlots, nameSuffix = nameSuffix)
    
    if saveLogs:
        
        saveLogs.flush()
        
    return(dataInd)


def checkNeighbourhood(df, indicator, printDetails = True, showPlots = True, savePlots = None, nameSuffix = '', saveLogs = None):
    
    """
    Check the consistency of values between two turning points,
    otherwise delete turning points that aren't the lowest/highest
    of neighbouring values.
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame (with one column), vector of values
    indicator: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes
    printDetails: bool
        print details about deleted extremes?
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot
    nameSuffix: str
        plot name suffix used when savePlots != None
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
        
    Returns
    -----
    indicator: pandas.DataFrame
        dataframe with extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
        
    """
    
    dataInd = indicator.copy()
    
    if printDetails:
        
        print('\nChecking extremes at %s for higher/lower neighbours:' % (dataInd.columns[0]))
    
    if saveLogs:
        
        saveLogs.write('\nChecking extremes at %s for higher/lower neighbours:' % (dataInd.columns[0]))
    
    lastDate = df.index[0]
    
    maxDate = dataInd.index[-1]
    
    try:
        
        thisDate = dataInd[dataInd != 0].first_valid_index()
        
    except:
        
        thisDate = maxDate
    
    while thisDate < maxDate:
        
        thisExt = dataInd.loc[thisDate][0]
        
        try:
        
            nextDate = dataInd[thisDate:][1:][dataInd != 0].first_valid_index()
            
        except IndexError:
            
            nextDate = maxDate
        
        if ((thisExt * df.loc[lastDate:nextDate]).max()[0] > (thisExt * df.loc[thisDate])[0]): # is there higher/lower point then this max/min?
            
            if printDetails:
                
                print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
            
            if saveLogs:
                
                saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
            
            dataInd.loc[thisDate] = 0
            
        else:
            
            lastDate = thisDate
            
        thisDate = nextDate
    
    if showPlots or savePlots:
        
        plotIndicator(df, dataInd, showPlots = showPlots, savePlots = savePlots, nameSuffix = nameSuffix)
    
    if saveLogs:
        
        saveLogs.flush()
        
    return(dataInd)


def checkCycleLength(df, indicator, cycleLength = 15, printDetails = True, showPlots = True, savePlots = None, nameSuffix = '', saveLogs = None):
    
    """
    Check the minimal length of cycle, otherwise delete one of the turning
    point (the lower/higher one for peaks/troughs).
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame (with one column), vector of values
    indicator: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes
    cycleLength: int
        minimal lenght of the cycle (in months)
    printDetails: bool
        print details about deleted extremes?
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot
    nameSuffix: str
        plot name suffix used when savePlots != None
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
        
    Returns
    -----
    indicator: pandas.DataFrame
        dataframe with extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
        
    """
    
    dataInd = indicator.copy()
    
    if printDetails:
        
        print('\nChecking extremes at %s for cycle length:' % (dataInd.columns[0]))
    
    if saveLogs:
        
        saveLogs.write('\nChecking extremes at %s for cycle length:' % (dataInd.columns[0]))
       
        
    for thisExt in [-1, 1]:
    
        if (dataInd[dataInd == thisExt].notnull().sum()[0] > 1): # more than 1 cycle?
            
            lastDate = dataInd[dataInd == thisExt].first_valid_index()
            thisDate = dataInd[dataInd == thisExt][lastDate:][1:].first_valid_index()
            
            while thisDate:
            
                realLength = dataInd[lastDate:thisDate].shape[0]
                
                if (realLength <= (cycleLength + 1)): # too short to be a cycle?
                
                    lastExt = thisExt # just to be very clear in the next lines
                    
                    if ((thisExt * df.loc[thisDate])[0] > (lastExt * df.loc[lastDate])[0]): # keep the higher one (or the earlier one when they equal)
                        
                        if printDetails:
                            
                            print('Deleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                        
                        if saveLogs:
                            
                            saveLogs.write('\nDeleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                        
                        dataInd.loc[lastDate] = 0
                        lastDate = thisDate
                        
                    else:
                        
                        if printDetails:
                        
                            print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                        
                        if saveLogs:
                            
                            saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                        
                        dataInd.loc[thisDate] = 0
                        
                else: 
                    
                    lastDate = thisDate
                    
                try:
                    
                    thisDate = dataInd[dataInd == thisExt][thisDate:][1:].first_valid_index()
                    
                except IndexError:
                    
                    break    
    
#    for i in range(len(dataInd)): # TODO: optimization, get rid of this for loop        
#        
#        thisExt = dataInd.iloc[i][0]
#        
#        if (thisExt != 0):
#            
#            rangeMin = max(0, i - cycleLength)
#            rangeMax = min(i + cycleLength + 1, len(dataInd))
#            
#            #print('Looking for other turning point at positions %d - %d.' % (rangeMin, rangeMax))
#            
#            thisDate = dataInd.index[i]
#            
##            if thisExt == 1:
##                
##                otherExt = dataInd.iloc[rangeMin:rangeMax].drop(thisDate).max()[0]
##                
##            else:
##                
##                otherExt = dataInd.iloc[rangeMin:rangeMax].drop(thisDate).min()[0]
#
#            otherExt = (thisExt * dataInd.iloc[rangeMin:rangeMax]).drop(thisDate).max()[0]
#            
#            #if otherExt != 0:
#            if otherExt == 1:
#                
##                if thisExt == 1:
##                    
##                    otherExtVal = df.iloc[rangeMin:rangeMax].drop(thisDate).max()[0]
##                    
##                else:
##                    
##                    otherExtVal = df.iloc[rangeMin:rangeMax].drop(thisDate).min()[0]
#
#                otherExtVal = (thisExt * df.iloc[rangeMin:rangeMax]).drop(thisDate).max()[0]
#                    
#                thisExtVal = thisExt * df.loc[thisDate][0]
#                
#                if otherExtVal > thisExtVal:
#                    
#                    dataInd.loc[thisDate] = 0
#                    
#                    print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
#                    
#                    if saveLogs:
#                        
#                            saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
    
    if showPlots or savePlots:
        
        plotIndicator(df, dataInd, showPlots = showPlots, savePlots = savePlots, nameSuffix = nameSuffix)
        
    if saveLogs:
        
        saveLogs.flush()
        
    return(dataInd)


def checkPhaseLength(df, indicator, phaseLength = 5, meanVal = 100, printDetails = True, showPlots = True, savePlots = None, nameSuffix = '', saveLogs = None):
    
    """
    Check the minimal length of phase, otherwise delete one of the turning
    points (the one which is less different from the mean).
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame (with one column), vector of values
    indicator: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes
    phaseLength: int
        minimal lenght of the phase (in months)
    meanVal: float
        mean value of the column, for series normalised by normaliseSeries() equals 100 (default)
    printDetails: bool
        print details about deleted extremes?
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot
    nameSuffix: str
        plot name suffix used when savePlots != None
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
        
    Returns
    -----
    indicator: pandas.DataFrame
        dataframe with extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
        
    """
        
    dataInd = indicator.copy()
    
    if printDetails:
        
        print('\nChecking extremes at %s for phase length:' % (dataInd.columns[0]))
    
    if saveLogs:
        
        saveLogs.write('\nChecking extremes at %s for phase length:' % (dataInd.columns[0]))
        
    
    if (dataInd[dataInd != 0].notnull().sum()[0] > 1): # more than 1 phase?
        
        lastDate = dataInd[dataInd != 0].first_valid_index()
        lastExt = dataInd.loc[lastDate][0]
        
        thisDate = dataInd[dataInd != 0][lastDate:][1:].first_valid_index()
        
        while thisDate:
            
            thisExt = dataInd.loc[thisDate][0]
            
            realLength = dataInd[lastDate:thisDate].shape[0]
            
            if (realLength <= (phaseLength + 1)): # too short to be a phase?
                
                lastVal = (df.loc[lastDate][0] - meanVal) * lastExt
                thisVal = (df.loc[thisDate][0] - meanVal) * thisExt
                
                if (thisVal > lastVal): # keep the one, which is deviated more (or the earlier one when they equal)
                    
                    if printDetails:
                        
                        print('Deleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                    
                    if saveLogs:
                        
                        saveLogs.write('\nDeleting extreme (%d) at %s' % (lastExt, str(lastDate)))
                    
                    dataInd.loc[lastDate] = 0
                    lastDate = thisDate
                    
                else:
                    
                    if printDetails:
                    
                        print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                    
                    if saveLogs:
                        
                        saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
                    
                    dataInd.loc[thisDate] = 0
                    
            else: 
                
                lastDate = thisDate
                lastExt = thisExt
                
            try:
                
                thisDate = dataInd[dataInd != 0][thisDate:][1:].first_valid_index()
                
            except IndexError:
                
                break
            
#    for i in range(len(dataInd)): # TODO: optimization, get rid of this for loop
#        
#        thisExt = dataInd.iloc[i][0]
#        
#        if (thisExt != 0):
#            
#            rangeMin = max(0, i - phaseLength)
#            
#            #print('Looking for other turning point at positions %d - %d.' % (rangeMin, i))
#            
#            thisDate = dataInd.index[i]
#
#            otherExt = (-1 * thisExt * dataInd.iloc[rangeMin:i]).max()[0]
#            
#            if otherExt == 1: # there is "oposite extreme"
#                    
#                dataInd.loc[thisDate] = 0
#                
#                print('Deleting extreme (%d) at %s' % (thisExt, str(thisDate)))
#                
#                if saveLogs:
#                    
#                    saveLogs.write('\nDeleting extreme (%d) at %s' % (thisExt, str(thisDate)))
    
    if showPlots or savePlots:
        
        plotIndicator(df, dataInd, showPlots = showPlots, savePlots = savePlots, nameSuffix = nameSuffix)
        
    if saveLogs:
        
        saveLogs.flush()
                
    return(dataInd)


#def deleteSideExtremes(df, indicator, showPlots = True, savePlots = None, nameSuffix = '', saveLogs = None):
#    """
#    OBSOLETE
#    
#    Use function checkNeighbourhood(), which deals also with higher/lower values between the turning points
#    
#    Delete first or last turning point, if there are higher/lower values in the beginning/end of the series.
#    
#    Parameters
#    -----
#    df:
#        pandas DataFrame (with one column), vector of values
#    indicator:
#        pandas DataFrame (with one column), vector of local extremes
#    showPlots:
#        show plots?
#    savePlots:
#        path where to save plot
#    nameSuffix:
#        plot name suffix used when savePlots != None
#    saveLogs:
#        file where to save stdouts (already opended with open())
#    """
#    
#    dataInd = indicator.copy()
#    
#    print('\nChecking extremes at %s for fake side turning points:' % (dataInd.columns[0]))
#    
#    if saveLogs:
#        
#        saveLogs.write('\nChecking extremes at %s for fake side turning points:' % (dataInd.columns[0]))
#    
#    firstDate = dataInd[dataInd != 0].first_valid_index()
#    
#    if firstDate:
#        
#        firstExt = dataInd.loc[firstDate][0]
#        firstExtVal = df.loc[firstDate][0]
#        
#        dataStart = df.loc[:firstDate].iloc[:-1]
#        
#        if ((firstExt * dataStart).max()[0] >= (firstExt * firstExtVal)):
#            
#            dataInd.loc[firstDate] = 0
#                    
#            print('Deleting extreme (%d) at %s' % (firstExt, str(firstDate)))
#            
#            if saveLogs:
#                
#                    saveLogs.write('\nDeleting extreme (%d) at %s' % (firstExt, str(firstDate)))
#
#    lastDate = dataInd[dataInd != 0].last_valid_index()
#    
#    if lastDate:
#        
#        lastExt = dataInd.loc[lastDate][0]
#        lastExtVal = df.loc[lastDate][0]
#        
#        dataEnd = df.loc[lastDate:].iloc[1:]
#        
#        if ((lastExt * dataEnd).max()[0] >= (lastExt * lastExtVal)):
#            
#            dataInd.loc[lastDate] = 0
#                    
#            print('Deleting extreme (%d) at %s' % (lastExt, str(lastDate)))
#            
#            if saveLogs:
#                
#                    saveLogs.write('\nDeleting extreme (%d) at %s' % (lastExt, str(lastDate)))
#
#    if showPlots or savePlots:
#        
#        plotIndicator(df, dataInd, savePlots = savePlots, nameSuffix = nameSuffix)
#        
#    if saveLogs:
#        
#        saveLogs.flush()
#        
#    return(dataInd)


def pipelineOneColumnTPDetection(col, printDetails = True, showPlots = True, savePlots = None, saveLogs = None, createInverse = False):
    
    """
    Pipeline connecting functions to detect turning points (local extremes,
    checking for alterations, checking for cycle and phase length).
    
    Parameters
    -----
    col: pandas.DataFrame
        monthly pandas DataFrame (series with index in format YYYY-MM) with one column
    printDetails: bool
        print details about deleted extremes?
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plots
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
    createInverse: bool
        create inverse time series?
        
    Returns
    -----
    indicator: pandas.DataFrame
        dataframe with extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
    indicatorInv: pandas.DataFrame
        dataframe with extremes of inverted series marked as -1 (troughs) or 1 (peaks)
        or 0 otherwise, returned, if createInverse = True
        
    """
    
    # a) Looking for local maxima/minima
    
    col_ind_local = getLocalExtremes(df = col, showPlots = showPlots, savePlots = savePlots, nameSuffix = '_04_localExt')
    
    
    # b) Check the turning points alterations
    
    col_ind_neigh = checkNeighbourhood(df = col, indicator = col_ind_local, printDetails = printDetails, showPlots = showPlots, saveLogs = saveLogs)
    col_ind_alter = checkAlterations(df = col, indicator = col_ind_neigh, printDetails = printDetails, showPlots = showPlots, saveLogs = saveLogs)
    
    
    # c) Check minimal length of cycle (15 months)
    
    col_ind_cycleLength = checkCycleLength(df = col, indicator = col_ind_alter, printDetails = printDetails, showPlots = showPlots, saveLogs = saveLogs)
    
    
    # d) Check the turning points alterations again
    
    col_ind_neighAgain = checkNeighbourhood(df = col, indicator = col_ind_cycleLength, printDetails = printDetails, showPlots = showPlots, saveLogs = saveLogs)
    col_ind_alterAgain = checkAlterations(df = col, indicator = col_ind_neighAgain, printDetails = printDetails, showPlots = showPlots, saveLogs = saveLogs)
    
    
    # e) Check minimal length of phase (5 months)
    
    col_ind_phaseLength = checkPhaseLength(df = col, indicator = col_ind_alterAgain, printDetails = printDetails, showPlots = showPlots, saveLogs = saveLogs)
    
    
    # f) Check the turning points alterations for the last time
    
    col_ind_neighLast = checkNeighbourhood(df = col, indicator = col_ind_phaseLength, printDetails = printDetails, showPlots = showPlots, saveLogs = saveLogs)
    col_ind_turningPoints = checkAlterations(df = col, indicator = col_ind_neighLast, printDetails = printDetails, showPlots = showPlots, savePlots = savePlots, nameSuffix = '_05_ext', saveLogs = saveLogs)  
    
    if createInverse:
        
        colName = col.columns[0]
        col_inv_ind_turningPoints = col_ind_turningPoints.copy() * -1
        col_inv_ind_turningPoints = col_inv_ind_turningPoints.rename(columns = {colName: str(colName) + '_INV'})
        
        return(col_ind_turningPoints, col_inv_ind_turningPoints)
        
    else:
        
        return(col_ind_turningPoints)


def pipelineTPDetection(df, origColumns = None, printDetails = True, showPlots = True, savePlots = None, saveLogs = None):    
    
    """
    Pipeline connecting functions to detect turning points (local extremes,
    checking for alterations, checking for cycle and phase length) for multiple
    column data frames.
    
    Parameters
    -----
    df: pandas.DataFrame
        monthly pandas DataFrame (series with index in format YYYY-MM),
        df can contain original series as well as inverted ones created
        by pipelineTransformations()
    origColumns: list or None
        list of original column names (not the inverted ones), if origColumns
        is provided, then the turning points are computed only once per each
        original + inverted pair of series, which leads to shorter computing
        time 
    printDetails: bool
        print details about deleted extremes?
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plots
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
        
    Returns
    -----
    indicator: pandas.DataFrame
        dataframe with extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
        
    """
    
    turningPoints = pd.DataFrame(index = df.index)
    
    if isinstance(origColumns, list):
        
        columns = origColumns
        createInverse = True
        
    else:
        
        columns = df.columns
        createInverse = False
    
    nCol = len(columns)
    
    for i, item in enumerate(columns):
        # i, item = list(enumerate(columns))[0]
        
        print("\nANALYSING SERIES %d from %d: %s" % (i + 1, nCol, item))
        
        if saveLogs:
            
            saveLogs.write("\n\nANALYSING SERIES %d from %d: %s" % (i + 1, nCol, item))
            saveLogs.flush()
        
        col = pd.DataFrame(df[item][df[item].notnull()], columns = [item])
        
        if (col.shape[0] == 0): # empty series
        
            print("\nWarning: Empty series.")
            
            if saveLogs:
                
                saveLogs.write("\nWarning: Empty series.")
                saveLogs.flush()
            
            col_turningPoints = pd.DataFrame(index = df.index, columns = [item])
            turningPoints = pd.concat([turningPoints, col_turningPoints], axis = 1)
            
            if createInverse:
                
                col_inv_turningPoints = pd.DataFrame(index = df.index, columns = [item + '_INV'])
                turningPoints = pd.concat([turningPoints, col_inv_turningPoints], axis = 1)
                
        else:
            
            # Get turning points
            
            if createInverse:
                
                col_turningPoints, col_inv_turningPoints = pipelineOneColumnTPDetection(col = col, printDetails = printDetails, showPlots = showPlots, savePlots = savePlots, saveLogs = saveLogs, createInverse = createInverse)
                
                if showPlots or savePlots:
                    
                    # Save also plot of inverse series (plot of original series saved during previous step)
                    
                    invColName = col_inv_turningPoints.columns[0]
                    plotIndicator(pd.DataFrame(df[invColName][df[invColName].notnull()]), col_inv_turningPoints, showPlots = showPlots, savePlots = savePlots, nameSuffix = '_05_ext')
                
                # Add to the DataFrame
                
                turningPoints = pd.concat([turningPoints, col_turningPoints, col_inv_turningPoints], axis = 1)
                
            else:
                
                col_turningPoints = pipelineOneColumnTPDetection(col = col, printDetails = printDetails, showPlots = showPlots, savePlots = savePlots, saveLogs = saveLogs, createInverse = createInverse)
                
                # Add to the DataFrame
                
                turningPoints = pd.concat([turningPoints, col_turningPoints], axis = 1)
            
    return(turningPoints)


def realTimeTPDetectionFromArchive(df, monthsToBeChecked = 3, indName = 'ind'):
    
    """
    Detect turning points from archive values of the series in real time.
    
    Parameters
    -----
    df: pandas.DataFrame
        monthly pandas DataFrame, downloaded from MEI_ARCHIVE or similar
        (each row is one month, each column is one edition of the same
        variable), the last six characters of each column name need to be 
        in '%Y%m' format (e.g., 200105)
    monthsToBeChecked: int
        how many consecutive months should be considered
    indName: str
        name of newly created indicator (default 'ind'); note: some functions
        (e.g., matchTurningPoints()) require that the names of indicators match
        the names of the series
        
    Returns
    -----
    realTime: pandas.DataFrame
        dataframe with extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
        (dates according to the month, when the extreme was found)
    foundAt: pandas.DataFrame
        dataframe with extremes marked as -1 (troughs) or 1 (peaks) or 0 otherwise
        (dates according to the edition, when the extreme was found)
        
    """
    
    dfShifted = df.shift(periods = 1)
    dfIndex = df/dfShifted
    
    firstMonth = None
    lastMonth = None
    firstEdition = None
    lastEdition = None
    
    realTime = pd.DataFrame(data = 0, index = df.index, columns = [indName])
    foundAt = pd.DataFrame(data = 0, index = df.index, columns = [indName])
    
    for i, item in enumerate(df.columns):
    
        col = pd.DataFrame(dfIndex[item], columns = [item])
        
        lastMonth = col.last_valid_index()
        monthsSelected = col.loc[(lastMonth - relativedelta(months = monthsToBeChecked - 1)) : lastMonth].copy()
        monthsSelected.dropna(inplace = True)
        
        if (firstMonth == None):
            
            firstMonth = lastMonth
        
        lastEdition = pd.to_datetime(item[-6:], format = '%Y%m')
        
        if (firstEdition == None):
            
            firstEdition = lastEdition
        
        if (monthsSelected.shape[0] == monthsToBeChecked): # this ignores non-complete data
            
            if ((monthsSelected > 1)[item].sum() == monthsToBeChecked): # growth in last n months
            
                realTime.loc[lastMonth, indName] = -1
                foundAt.loc[lastEdition, indName] = -1
                
            elif ((monthsSelected < 1)[item].sum() == monthsToBeChecked): # decline in last n months
                
                realTime.loc[lastMonth, indName] = 1
                foundAt.loc[lastEdition, indName] = 1
    
    lastSeries = pd.DataFrame(df.iloc[ : , -1])
    
    realTime = checkAlterations(lastSeries, realTime, keepFirst = True, showPlots = False)[firstMonth : lastMonth]
    foundAt = checkAlterations(lastSeries, foundAt, keepFirst = True, showPlots = False)[firstEdition : lastEdition]
    
    # Delete false extremes at the beginning of the series
    
    realTime.iloc[0] = 0
    foundAt.iloc[0] = 0
    
    return(realTime, foundAt)


# TURNING-POINT MATCHING & EVALUATION

def matchTurningPoints(ind1, ind2, lagFrom = -9, lagTo = 24, printDetails = True, saveLogs = None):
    
    """
    Compare turning points of reference and idividual time series. 
    
    Parameters
    -----
    ind1: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes of reference
        series (gold standard)
    ind2: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes of second series
    lagFrom: int
        minimal lag where to look for the match, default -9 (9 months lag)
    lagTo: int
        maximal lag where to look for the match, default +24 (24 months lead)
    printDetails: bool
        print details about deleted extremes?
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
        
    Returns
    -----
    extOrd: pandas.DataFrame
        dataframe with order of matched turning points
    time: pandas.DataFrame
        dataframe with time differences (in months) between the matched turning points
    missing: pandas.DataFrame
        dataframe with marked missing turning points
    missingEarly: pandas.DataFrame
        dataframe with marked early missing turning points
    extra: pandas.DataFrame
        dataframe with marked false signals
        
    """
    
    # TODO:
    # Create logs?
    
    # Input check
    
    if (lagTo <= lagFrom):
        
        if printDetails:
            
            print("Error: parameter lagTo should be higher than parameter lagFrom.")
        
        if saveLogs:
            
            saveLogs.write("\nError: parameter lagTo should be higher than parameter lagFrom.")
            saveLogs.flush()
            
        return(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    
    
    # Matching turning points
    
    refName = ind1.columns[0]
    indName = ind2.columns[0]
    
    dataExt = ind1.loc[ind1[refName] != 0].copy()
    dataExt['extOrd'] = range(len(dataExt))
    
    refIndexes = ind2.index[(ind2.index).isin(ind1.index)] # same time period as reference series 
    
    dataInd = ind2.loc[refIndexes].copy()
    
    if (len(refIndexes) == 0):
        
        if printDetails:
            
            print("Warning: There is no overlapping period in the time series.")
            
        if saveLogs:
            
            saveLogs.write("\nWarning: There is no overlapping period in the time series.")
            
        refIndexes = ind1.index
        dataInd = pd.DataFrame(0, columns = [indName], index = refIndexes)
    
    dataInd['extOrd'] = np.nan
    dataInd['time'] = np.nan
    dataInd['missing'] = np.nan # only turning points that could be in this time series (from the beginning of this series to the end of the reference series)
    dataInd['missingEarly'] = np.nan # turning points that occured in the reference series before this time series started
    dataInd['extra'] = np.nan
        
    if len(dataExt) > 0:
        
        shiftedIndex = pd.date_range(start = min(ind2.index) + relativedelta(months = lagFrom)
                                        , end = max(ind2.index) + relativedelta(months = lagTo), freq = 'MS')
        ind2Shifted = pd.DataFrame(index = shiftedIndex, data = ind2)
        dataShifted = pd.DataFrame(index = shiftedIndex)
        
        for i in range(lagFrom, lagTo): # plus means lead, minus means lag
            
            dataShifted = pd.concat([dataShifted, ind2Shifted.shift(i).rename(columns = {indName: 'shift_' + str(i)})], axis = 1)
        
        for date, row in dataExt.iterrows():
            #date, row = list(dataExt.iterrows())[19]
            
            thisExt = row.loc[refName]
            thisExtOrd = row.loc['extOrd']
            
            try:
                
                dataShiftedThisDate = dataShifted.loc[date]
                
                possibleExt = pd.DataFrame(dataShiftedThisDate[dataShiftedThisDate == thisExt])
                
                if (len(possibleExt) == 0):
                        
                    if (date >= min(refIndexes)): # this turning point could be in the series
                        
                        dataInd.loc[date, ['missing']] = True
                    
                    else:
                        
                        if printDetails:
                            
                            print("Warning: Missing cycle caused by short series (early turning point).")
                        
                        if saveLogs:
                            
                            saveLogs.write("\nWarning: Missing cycle caused by short series (early turning point).")
                            
                        dataInd.loc[date, ['missingEarly']] = True
                    
                else:
                    
                    shifts = [int(i[6:]) for i, j in possibleExt.iterrows()]
                    minShift = min(shifts, key = abs)
                    
                    dateShift = date - relativedelta(months = minShift)
                    
                    existingOrd = dataInd.loc[dateShift, 'extOrd']
                    
                    if (not(np.isnan(existingOrd))): # peak/trough is already occupied
                        
                        existingTime = dataInd.loc[dateShift, 'time']
                        
                        if (abs(existingTime) > abs(minShift)): # new peak/trough is closer
                            
                            if printDetails:
                                
                                print("Warning: Turning point at %s already matched, changing now from order %d to %d." % (dateShift.strftime("%Y-%m-%d"), existingOrd, thisExtOrd))
                            
                            if saveLogs:
                            
                                saveLogs.write("\nWarning: Turning point at %s already matched, changing now from order %d to %d." % (dateShift.strftime("%Y-%m-%d"), existingOrd, thisExtOrd))
                            
                            existingOrdDate = dataExt[dataExt['extOrd'] == existingOrd].index
                            dataInd.loc[existingOrdDate, ['missing']] = True
                        
                            dataInd.loc[dateShift, 'time'] = minShift
                            dataInd.loc[dateShift, 'extOrd'] = thisExtOrd
                        
                            # TODO: maybe there were more possibilities, how to match
                            # the existing turning point, so they should be revised? 
                            
                        else: # new peak/trough is further then the existing one
                            
                            dataInd.loc[date, ['missing']] = True
                    
                    else: # empty spot
                        
                        dataInd.loc[dateShift, 'time'] = minShift
                        dataInd.loc[dateShift, 'extOrd'] = thisExtOrd
                    
                    
            except KeyError:
                
                if (date >= min(refIndexes)): # this turning point could be in the series
                    
                    if printDetails:
                        
                        print("Warning: Missing cycle caused by short series (regular turning point).")
                    
                    if saveLogs:
                            
                        saveLogs.write("\nWarning: Missing cycle caused by short series (regular turning point).")
                    
                    dataInd.loc[date, ['missing']] = True
                    
                else:
                    
                    if printDetails:
                        
                        print("Warning: Missing cycle caused by short series (early turning point).")
                    
                    if saveLogs:
                            
                        saveLogs.write("\nWarning: Missing cycle caused by short series (early turning point).")
                    
                    dataInd.loc[date, ['missingEarly']] = True
        
    else:
        
        if printDetails:
            
            print("Warning: There are no turning points in the reference series.")
            
        if saveLogs:
                            
            saveLogs.write("\nWarning: There are no turning points in the reference series.")
    
    
    dataInd.sort_index(inplace = True)
    
    
    # Check order of turning points
    
    lastOrder = 0
    lastTime = None
    lastDate = None
    
    for thisDate, row in dataInd[dataInd['extOrd'].notnull()].iterrows():
        #thisDate, row = list(dataInd[dataInd['extOrd'].notnull()].iterrows())[6]
        
        #print('Check %s' % (thisDate))
        
        thisOrder = row['extOrd']
        thisTime = row['time']
        
        if (thisOrder < lastOrder): 
            
            if printDetails:
            
                print("Warning: Discrepancy between order of turning points %s and %s." % (lastDate.strftime("%Y-%m-%d"), thisDate.strftime("%Y-%m-%d")))
            
            if saveLogs:
                            
                saveLogs.write("\nWarning: Discrepancy between order of turning points %s and %s." % (lastDate.strftime("%Y-%m-%d"), thisDate.strftime("%Y-%m-%d")))
            
            if (abs(thisTime) < abs(lastTime)): # keep the one which is closer to the turning point
                
                if printDetails:
                    
                    print("<-- %s deleted from matched turning points." % lastDate.strftime("%Y-%m-%d"))
                    
                if saveLogs:
                            
                    saveLogs.write("\n<-- %s deleted from matched turning points." % lastDate.strftime("%Y-%m-%d"))
                    
                dataInd.loc[lastDate, 'extOrd'] = np.nan
                dataInd.loc[lastDate, 'time'] = np.nan
                
                lastOrdDate = dataExt[dataExt['extOrd'] == lastOrder].index
                dataInd.loc[lastOrdDate[0], ['missing']] = True
                
                lastOrder = thisOrder
                lastTime = thisTime
                lastDate = thisDate
                
            else:
                
                if printDetails:
                    
                    print("<-- %s deleted from matched turning points." % thisDate.strftime("%Y-%m-%d"))
                    
                if saveLogs:
                            
                    saveLogs.write("\n<-- %s deleted from matched turning points." % thisDate.strftime("%Y-%m-%d"))
                
                dataInd.loc[thisDate, 'extOrd'] = np.nan
                dataInd.loc[thisDate, 'time'] = np.nan
                
                thisOrdDate = dataExt[dataExt['extOrd'] == thisOrder].index
                dataInd.loc[thisOrdDate[0], ['missing']] = True
                
        else:
            
            lastOrder = thisOrder
            lastTime = thisTime
            lastDate = thisDate
            
        
        #print(thisOrder)
        #print(thisTime)
        #print(thisDate)
    
    
    # Mark extras
    
    dataInd.loc[((dataInd[indName] != 0) 
        & (dataInd[indName].notnull())
        & (dataInd['extOrd'].isnull()))
        , 'extra'] = True
    
    
    # Check last extremes    
    # Did the last extreme occur in the last n months (n = lagTo) of the reference
    # series and was it an extra one? -->
    # Such extreme shouldn't be marked as extra, as there is still a chance,
    # that the "true extreme" would occur.
    
    lastExt = dataInd['extOrd'].last_valid_index()
    lastExtra = dataInd['extra'].last_valid_index()
    
    if (
        (
             lastExtra != None
             and
             (lastExtra > (ind1.last_valid_index() - relativedelta(months = lagTo)))
        )
        and
        (
             lastExt == None
             or 
             (lastExtra > lastExt)
        )
    ):
            
        if printDetails:
            
            print("Warning: Last extreme wasn\'t marked as extra, because it was too close to the end of reference series.")
        
        if saveLogs:
                            
            saveLogs.write("\nWarning: Last extreme wasn\'t marked as extra, because it was too close to the end of reference series.")
        
        dataInd.loc[lastExtra, 'extra'] = np.nan
    
    
    if saveLogs:
        
        saveLogs.flush()
    
    
    # Return results
    
    return(pd.DataFrame(dataInd['extOrd']).rename(columns = {'extOrd': indName})
        , pd.DataFrame(dataInd['time']).rename(columns = {'time': indName})
        , pd.DataFrame(dataInd['missing']).rename(columns = {'missing': indName})
        , pd.DataFrame(dataInd['missingEarly']).rename(columns = {'missingEarly': indName})
        , pd.DataFrame(dataInd['extra']).rename(columns = {'extra': indName}))


def pipelineTPMatching(df1, df2, ind1, ind2, printDetails = True, showPlots = True, savePlots = None, nameSuffix = '_06_matching', saveLogs = None, bw = False, lagFrom = -9, lagTo = 24):
    
    """
    Pipeline to compare turning points of reference and idividual time series.
    
    Parameters
    -----
    df1: pandas.DataFrame
        pandas DataFrame (with one column), values of reference series (gold standard)
    df2: pandas.DataFrame
        pandas DataFrame, values of individual economic series
    ind1: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes of reference
        series (gold standard), the data frame needs to have the same column names
        as the df1
    ind2: pandas.DataFrame
        pandas DataFrame, vector of turning points of individual economic series,
        the data frame needs to have the same column names as the df2
    printDetails: bool
        print details about deleted extremes?
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plots
    nameSuffix: str
        plot name suffix used when savePlots != None
    saveLogs: _io.TextIOWrapper or None
        file where to save stdouts (already opended with open())
    bw: bool
        plot in black and white, default is False
    lagFrom: int
        minimal lag where to look for the match, default -9 (9 months lag)
    lagTo: int
        maximal lag where to look for the match, default +24 (24 months lead)
        
    Returns
    -----
    extOrd: pandas.DataFrame
        dataframe with order of matched turning points
    time: pandas.DataFrame
        dataframe with time differences (in months) between the matched turning points
    missing: pandas.DataFrame
        dataframe with marked missing turning points
    missingEarly: pandas.DataFrame
        dataframe with marked early missing turning points
    extra: pandas.DataFrame
        dataframe with marked false signals
    """
    
    extOrd = pd.DataFrame(index = df2.index)
    time = pd.DataFrame(index = df2.index)
    missing = pd.DataFrame(index = df2.index)
    missingEarly = pd.DataFrame(index = df2.index)
    extra = pd.DataFrame(index = df2.index)
    
    nCol = df2.shape[1]
    
    for i, item in enumerate(df2.columns):
        # i, item = list(enumerate(df2.columns))[0]
        
        print("\nANALYSING SERIES %d from %d: %s" % (i + 1, nCol, item))
        
        if saveLogs:
            
            saveLogs.write("\n\nANALYSING SERIES %d from %d: %s" % (i + 1, nCol, item))
            saveLogs.flush()
        
        col = pd.DataFrame(df2[item][df2[item].notnull()], columns = [item])
        col_ind = pd.DataFrame(ind2[item][ind2[item].notnull()], columns = [item])
        
        
        if (col.shape[0] == 0): # empty series
            
            if printDetails:
                
                print("\nWarning: Empty series.")
            
            if saveLogs:
                
                saveLogs.write("\nWarning: Empty series.")
                saveLogs.flush()
            
            col_extOrd = pd.DataFrame(index = df2.index, columns = [item])
            col_time = pd.DataFrame(index = df2.index, columns = [item])
            col_missing = pd.DataFrame(index = df2.index, columns = [item])
            col_missingEarly = pd.DataFrame(index = df2.index, columns = [item])
            col_extra = pd.DataFrame(index = df2.index, columns = [item])
        
        else:
            
            # Match turning points
            
            col_extOrd, col_time, col_missing, col_missingEarly, col_extra = matchTurningPoints(ind1 = ind1, ind2 = col_ind, printDetails = printDetails, saveLogs = saveLogs)
            
            
            # Plot turning points
            
            if (showPlots or savePlots):
                
                compareTwoIndicators(df1, col, ind1, col_ind, col_extOrd, showPlots = showPlots, savePlots = savePlots, nameSuffix = nameSuffix, bw = bw)
            
        
        # Add to the DataFrame
        
        extOrd = pd.concat([extOrd, col_extOrd], axis = 1)
        time = pd.concat([time, col_time], axis = 1)
        missing = pd.concat([missing, col_missing], axis = 1)
        missingEarly = pd.concat([missingEarly, col_missingEarly], axis = 1)
        extra = pd.concat([extra, col_extra], axis = 1) 
    
    
    # Return results
    
    return(extOrd, time, missing, missingEarly, extra)


def crossCorrelation(df1, df2, lagFrom = -9, lagTo = 24):
    
    """
    Compute cross correlations and returns the highest one and its position.
    
    Parameters
    -----
    df1: pandas.DataFrame
        pandas DataFrame (with one column), values of reference series (gold standard)
    df2: pandas.DataFrame
        pandas DataFrame (with one column), values of second series with different name than the first one
    lagFrom: int
        minimal lag of cross correlations, default -9 (9 months lag)
    lagTo: int
        maximal lag of cross correlations, default +24 (24 months lead)
        
    Returns
    -----
    corrMax: pandas.DataFrame
        dataframe with maximal value of cross correlations
    corrPos: pandas.DataFrame
        dataframe with position of cross corralation maximum
    """
    
    # Input check
    
    if (lagTo <= lagFrom):
        
        print('Error: parameter lagTo should be higher than parameter lagFrom.')
        return(None, None)
    
    
    # Cross correlations
    
    corrMe = pd.concat([df1.rename(columns = {df1.columns[0]: 'ref'})
                        , df2.rename(columns = {df2.columns[0]: 'ind'})
                        ]
                       , axis = 1).astype('float64')
    corrMe.dropna(how = 'all', inplace = True)
    
    corrMax = 0
    corrPos = np.nan
    
    for i in range(lagFrom, lagTo):
        
        corrThis = corrMe['ref'].corr(corrMe['ind'].shift(i))
        
        #print('Shift: %d, correlation: %f' % (i, corrThis))
        
        if (corrThis > corrMax):
            
            corrMax = corrThis
            corrPos = i
            
            #print('New and better value!')
            
    return(corrMax, corrPos)


def pipelineEvaluation(df1, df2, missing, missingEarly, extra, time, checkCorr = True, maxInd = None, evalOnly = False, weights = [0.25, 0.05, 0.15, 0.15, 0.00, 0.10, 0.15, 0.15]):
    
    """
    Pipeline to choose the best individual series for composite leading indicator (computing
    number of missing turning points (regular and early), number of extra turning points,
    mean lead time, median lead time, standard deviation of lead time, coefficient of variation
    of lead time, maximum of correlation coefficient, position of maximum of correlation
    coefficient, sanity check (= difference between position of maximum of correlation
    coefficient and median lead time)). With evalOnly = False, the weights are added
    to each of these criteria to rank the individual series and select the best.
       
    Parameters
    -----
    df1: pandas.DataFrame
        pandas DataFrame (with one column), values of reference series (gold standard)
    df2: pandas.DataFrame
        pandas DataFrame, individual indicators to be compared with reference series
    missing: pandas.DataFrame
        pandas DataFrame, missing turning points indicators (result of matchTurningPoints())
    missingEarly: pandas.DataFrame
        pandas DataFrame, missing early turning points indicators (result of matchTurningPoints())
    extra: pandas.DataFrame
        pandas DataFrame, extra turning points indicators (result of matchTurningPoints())
    time: pandas.DataFrame
        pandas DataFrame, time of the turning points indicators (result of matchTurningPoints())
    maxInd: int or None
        how many indicators should be returned at most (default None returns all that pass the conditions)?
    checkCorr: bool
        should the highly correlated individual series be ignored (default True)?
    evalOnly: bool
        if True, return only evaluation matrix; if False (default), return evaluation matrix
        with added total column (total rank), evaluation matrix of selected indicators and
        vector of selected columns
    weights: list
        weigths of 8 criteria:
            
        - number of missing turning points
        - number of missing early turning points
        - number of extra turning points
        - mean lead time
        - standard deviation of lead time
        - coefficient of variation of lead time
        - sanity check
        - maximum of correlation coefficient
        
        the sum of these weights should be equal to 1 for easier interpretation of the results,
        but this is not necessary; weights parameter is ignored when evalOnly = True
        
    Returns
    -----        
    totalEval: pandas.DataFrame
        dataframe with evaluation metrics of all series
    selectedEval: pandas.DataFrame
        dataframe with evaluation metrics of selected series, returned, if
        evalOnly = False
    selectedCol: pandas.indexes.base.Index
        names of selected series, returned, if evalOnly = False
        
    """
    
    # a) Check of input values
    
    if (len(weights) != 8):
        
        print("Error: Wrong number of weights (8 expected). For more details see function description.")
        
        return(None, None, None)
    
    
    # b) Basic characteristics
    
    df_eval = pd.DataFrame(index = df2.columns)
    
    df_eval['targeted'] = missing.sum().fillna(0) +  missingEarly.sum().fillna(0) + time.notnull().sum().fillna(0)
    df_eval['missing'] = missing.sum().fillna(0)
    df_eval['missingEarly'] = missingEarly.sum().fillna(0)
    df_eval['extra'] = extra.sum().fillna(0)
    df_eval['leadMean'] = time.mean().fillna(0)
    df_eval['leadMedian'] = time.median().fillna(0)
    df_eval['leadStDev'] = time.std().fillna(0)
    df_eval['leadCVar'] = abs(df_eval['leadStDev']/df_eval['leadMean'])
    
    nCol = df2.shape[1]
    
    for i, item in enumerate(df2.columns):
        
        print("\nANALYSING SERIES %d from %d: %s" % (i + 1, nCol, item))
        
        col = pd.DataFrame(df2[item][df2[item].notnull()], columns = [item])
        
        col_corrMax, col_corrPos = crossCorrelation(df1 = df1, df2 = col)
        
        df_eval.loc[col.columns[0], 'corrMax'] = col_corrMax
        df_eval.loc[col.columns[0], 'corrPosition'] = col_corrPos
        
    df_eval['sanityCheck'] = abs(df_eval['corrPosition'] - df_eval['leadMedian'])
    
    if evalOnly:
        
        return(df_eval)
        
    
    # c) Selection of eligible indicators
    
    df_evalWeights = pd.DataFrame(index = df2.columns)
    
    df_evalWeights['missing'] = df_eval['missing'].rank(method = 'max', na_option = 'keep', ascending = False)
    df_evalWeights['missingEarly'] = df_eval['missingEarly'].rank(method = 'max', na_option = 'keep', ascending = False)
    df_evalWeights['extra'] = df_eval['extra'].rank(method = 'max', na_option = 'keep', ascending = False)
    df_evalWeights['leadMean'] = df_eval['leadMean'].abs().rank(method = 'max', na_option = 'keep', ascending = True) # note abs value!!!
    df_evalWeights['leadStDev'] = df_eval['leadStDev'].rank(method = 'max', na_option = 'keep', ascending = False)
    df_evalWeights['leadCVar'] = df_eval['leadCVar'].rank(method = 'max', na_option = 'keep', ascending = False)
    df_evalWeights['sanityCheck'] = df_eval['sanityCheck'].rank(method = 'max', na_option = 'keep', ascending = False)
    df_evalWeights['corrMax'] = df_eval['corrMax'].rank(method = 'max', na_option = 'keep', ascending = True)
    
    df_evalWeights['total'] = (
            weights[0] * df_evalWeights['missing'] +
            weights[1] * df_evalWeights['missingEarly'] +
            weights[2] * df_evalWeights['extra'] +
            weights[3] * df_evalWeights['leadMean'] +
            weights[4] * df_evalWeights['leadStDev'] +
            weights[5] * df_evalWeights['leadCVar'] +
            weights[6] * df_evalWeights['sanityCheck'] +
            weights[7] * df_evalWeights['corrMax']
            )
    
    df_totalEval = pd.concat([df_eval, df_evalWeights['total']], axis = 1)
    df_selectedEval = df_totalEval[
        #(df_totalEval['leadMean'] > 3)
        (df_totalEval['leadMedian'] >= 3)
        #& (df_totalEval['corrMax'] > 0.5)
        & (df_totalEval['corrPosition'] >= 0)
        & (df_totalEval['total'] > 0.85 * df_totalEval['total'].max())
        ].copy()
    df_selectedEval.sort_values(by = 'total', ascending = False, inplace = True)
    
    if checkCorr:
        
        print('\nChecking correlated indicators.')
        
        dataCorr = df2[df_selectedEval.index].copy()
        
        cMat = (abs(dataCorr.corr(method = 'pearson')) >= 0.99).astype('int') # correlation
        cMat = pd.DataFrame(np.tril(cMat, k = -1), index = cMat.index, columns = cMat.index) # sum under diagonal
        
        highCorr = cMat[cMat.sum(axis = 1) != 0].index
                        
        print('\n%d indicators aren\'t considered because of high correlation:' % len(highCorr))
        print('\n'.join(highCorr))
        
        df_selectedEval = df_selectedEval.loc[~df_selectedEval.index.isin(highCorr)]
    
    if maxInd and (maxInd < df_selectedEval.shape[0]):
        
        print('Returning only subset (%d) of eligible indicators.' % (maxInd))
        
        df_selectedEval = df_selectedEval[:maxInd]
        
    df_selectedCol = df_selectedEval.index
    
    return(df_totalEval, df_selectedEval, df_selectedCol)


# 7) AGGREGATION

def pipelineCreateCLI(df):
    
    """
    Pipeline to compute composite indator from selected individual time series.
    
    Parameters
    -----
    df: pandas.DataFrame
        pandas DataFrame of transformed series (seasonally adjusted, detrended
        and normalised), with one column per each selected indicator
        
    Returns
    -----        
    CLI: pandas.DataFrame
        dataframe with created composite indicator
        
    """
    
    # a) Chain linking
    
    cMat = df
    
    wMat = pd.DataFrame(data = 1 / cMat.shape[1]
        , index = cMat.index
        , columns = cMat.columns) # weight of indicator (uniform)
    
    deltaMat = (cMat * cMat.shift(1)).notnull().astype(int) # binary indicator if component i is available in both period t and t - 1
    
    mult = (wMat * deltaMat * cMat).sum(axis = 1) / (wMat * deltaMat * cMat.shift(1)).sum(axis = 1)
    
    elig = (wMat * deltaMat).sum(axis = 1) > 0.6 # eligible time periods
    
    
    # b) Composite indicator
    
    CLI = pd.DataFrame(index = elig[elig == True].index, columns = ['CLI'])
    
    CLI.iloc[0] = 100
    
    for i in range(1, len(CLI)):
        
        thisDate = CLI.index[i]    
        CLI.iloc[i] = mult.loc[thisDate] * CLI.iloc[i - 1]
        
    return(CLI)      


# VISUALISATIONS

def plotHP(data, phase = 1):
    
    """
    Plot outputs from statsmodels Hodrick-Prescott filter.
    
    Parameters
    -----
    data: pandas.DataFrame
        output from statsmodels Hodrick-Prescott filter
    phase: int
        first or second application of the filter (default = 1)
    """
    
    if phase == 1:
        
        str1 = 'Trend component'
        str2 = 'Original series'
        str3 = 'Cyclical component (with noise)'
        plotTitle = 'Hodrick-Prescott filter (first application)'
#        col1 = 'black'
#        col2 = 'gray'
#        col3 = 'gray'
        
    else:
        
        str1 = 'Cyclical component'
        str2 = 'Cyclical component (with noise)'
        str3 = 'Noise'
        plotTitle = 'Hodrick-Prescott filter (second application)'
#        col1 = 'gray'
#        col2 = 'black'
#        col3 = 'gray'
    
    fontP = mpl.font_manager.FontProperties()
    fontP.set_size('xx-small')
    
    plt.figure(1)
    plt.subplot(211)
    plt.title(plotTitle)
    plt.plot(data[1] + data[0], label = str2, color = 'gray')
    plt.plot(data[1], label = str1, color = 'black')
    plt.legend(loc = 'best'
        , prop = fontP
        , frameon = False)
    
    plt.subplot(212)
    plt.plot(data[0], label = str3, color = 'gray')
    plt.legend(loc = 'best'
        , prop = fontP
        , frameon = False)
    plt.show()


def compareTwoSeries(df1, df2):
    
    """
    Plot two series in one plot, first on left axis, second on rigth axis.
    
    Parameters
    -----
    df1: pandas.DataFrame
        pandas DataFrame (with one column)
    df2: pandas.DataFrame
        pandas DataFrame (with one column)
    """
    
    plotMe = pd.concat([df1, df2], axis = 1)
    plotMe.dropna(how = 'all', inplace = True)
    plotMe['time'] = range(len(plotMe))
    
    col1 = plotMe.columns[0]
    col2 = plotMe.columns[1]
    
    if (len(plotMe) < 12*25): 
        
        data_values = [row['time'] for index, row in plotMe.iterrows() if index.month == 1]
        data_labels = [index.strftime('%Y-%m') for index, row in plotMe.iterrows() if index.month == 1]
    
    else:
        
        data_values = [row['time'] for index, row in plotMe.iterrows() if (index.month == 1 & (index.year % 5 == 0))]
        data_labels = [index.strftime('%Y-%m') for index, row in plotMe.iterrows() if (index.month == 1 & (index.year % 5 == 0))]
    
    fig, ax1 = plt.subplots()
    
    plt.xticks(data_values, data_labels, rotation = 60)
    
    ax1.plot(plotMe[-plotMe[col1].isnull()]['time'], plotMe[-plotMe[col1].isnull()][col1], linestyle = '-', color = 'gray')
    ax2 = ax1.twinx()
    ax2.plot(plotMe[-plotMe[col2].isnull()]['time'], plotMe[-plotMe[col2].isnull()][col2], linestyle = '-', color = 'black')
    
    fig.tight_layout() 
    plt.show()
    

def plotIndicator(df1, df2, showPlots = True, savePlots = None, namePrefix = '', nameSuffix = ''):
    
    """
    Plot series and vertical lines for not null indicator values.
    
    Parameters
    -----
    df1: pandas.DataFrame
        pandas DataFrame (with one column)
    df2: pandas.DataFrame
        pandas DataFrame (with one column, which is used to indicate vertical lines)
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot
    namePreffix: str
        plot name prefix used when savePlots != None
    nameSuffix: str
        plot name suffix used when savePlots != None
    """
    
    if not(showPlots or savePlots):
        
        print('Warning: There is no point in running compareTwoIndicators() without either showing or saving the plots!')
        
    
    if df1.columns[0] == df2.columns[0]:
        
        plotMe = pd.concat([df1, df2.rename(columns = {df2.columns[0]: 'ind'})], axis = 1)
        
    else:
        
        plotMe = pd.concat([df1, df2], axis = 1)
        
    plotMe.dropna(how = 'all', inplace = True)
    plotMe['time'] = range(len(plotMe))
    
    col1 = plotMe.columns[0]
    col2 = plotMe.columns[1]
    
    if (len(plotMe) < 12*25): 
        
        data_values = [row['time'] for index, row in plotMe.iterrows() if index.month == 1]
        data_labels = [index.strftime('%Y-%m') for index, row in plotMe.iterrows() if index.month == 1]
    
    else:
        
        data_values = [row['time'] for index, row in plotMe.iterrows() if (index.month == 1 & (index.year % 5 == 0))]
        data_labels = [index.strftime('%Y-%m') for index, row in plotMe.iterrows() if (index.month == 1 & (index.year % 5 == 0))]
    
    plt.ioff() # Turn interactive plotting off
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 2.5))
    
    plt.xticks(data_values, data_labels, rotation = 60)
    
    ax.plot(plotMe[-plotMe[col1].isnull()]['time'], plotMe[-plotMe[col1].isnull()][col1], linestyle = '-', color = 'gray')
    
    
    # Vertical lines
    
    for i, row in plotMe.iterrows(): # TODO: zbavit se for cyklu jako v compareTwoIndicators()
        
        if row[col2] == 1:
            
            plt.axvline(x = row['time'], linestyle = '--', color = 'black')
            
        elif row[col2] == -1:
            
            plt.axvline(x = row['time'], linestyle = '-.', color = 'black')
    
    fig.tight_layout() 
    
    if savePlots:
        
        fig.savefig(os.path.join(savePlots, namePrefix + str(col1) + nameSuffix), dpi = 300)
    
    if showPlots:
        
        plt.show()
    
    plt.close(fig)


def compareTwoIndicators(df1, df2, ind1, ind2, ord2, showPlots = True, savePlots = None, namePrefix = '', nameSuffix = '', bw = False):
    
    """
    Plot reference series with turning points and compare it with turning points of second time series.
    
    Parameters
    -----
    df1: pandas.DataFrame
        pandas DataFrame (with one column), values of reference series (gold standard)
    df2: pandas.DataFrame
        pandas DataFrame (with one column), values of second series
    ind1: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes of reference series (gold standard)
    ind2: pandas.DataFrame
        pandas DataFrame (with one column), vector of local extremes of second series
    ord2: pandas.DataFrame
        pandas DataFrame (with one column), orders of local extremes of second series
    showPlots: bool
        show plots?
    savePlots: str or None
        path where to save plot
    namePreffix: str
        plot name prefix used when savePlots != None
    nameSuffix: str
        plot name suffix used when savePlots != None
    bw: bool
        plot in black and white, default is False
    """
    
    if not(showPlots or savePlots):
        
        print('Warning: There is no point in running compareTwoIndicators() without either showing or saving the plots!')
    
    plotMe = pd.concat([df1.rename(columns = {df1.columns[0]: 'first'})
                        , df2.rename(columns = {df2.columns[0]: 'second'})
                        , ind1.rename(columns = {ind1.columns[0]: 'firstInd'})
                        , ind2.rename(columns = {ind2.columns[0]: 'secondInd'})
                        , ord2.rename(columns = {ord2.columns[0]: 'peakOrd'})]
                        , axis = 1)
    plotMe = plotMe[plotMe['first'].notnull()]
    plotMe['time'] = range(len(plotMe))
    
    if (len(plotMe) < 12*25): # shorter then 25 years --> one label per each year
        
        data_values = [row['time'] for index, row in plotMe.iterrows() if index.month == 1]
        data_labels = [index.strftime('%Y-%m') for index, row in plotMe.iterrows() if index.month == 1]
    
    else: # 25 years and longer --> one label per each 5 years
        
        data_values = [row['time'] for index, row in plotMe.iterrows() if (index.month == 1 & (index.year % 5 == 0))]
        data_labels = [index.strftime('%Y-%m') for index, row in plotMe.iterrows() if (index.month == 1 & (index.year % 5 == 0))]
    
    df1_ext = plotMe[plotMe['firstInd'] != 0]
    
    numColors = len(df1_ext)
    
    if bw:
        
        currentColors = ['lightgray' for i in range(numColors)]
        
    else:
        
        cmap = mpl.cm.get_cmap(name = 'rainbow')
        currentColors = [cmap(1.*i/numColors) for i in range(numColors)]
    
    plt.ioff() # Turn interactive plotting off
    
    fig, (ax1, ax2) = plt.subplots(2, sharex = True)
    
    plt.xticks(data_values, data_labels, rotation = 60)
    
    # reference series
    
    ord1 = 0
    ax1.plot(plotMe['time'], plotMe['first'], linestyle = '-', color = 'gray')
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    for i, row in df1_ext.iterrows():
        
        c = currentColors[ord1]
        
        if row['firstInd'] == 1:
            
            ls = '--'
        
        else:
            
            ls = '-.'
            
        ax1.axvline(x = row['time'], linestyle = ls, color = c)
        
        ord1 += 1
    
    
    # individual series
            
    ax2.plot(plotMe['time'], plotMe['second'], linestyle = '-', color = 'gray')
    
    for i, row in plotMe[(plotMe['secondInd'] == -1) | (plotMe['secondInd'] == 1)].iterrows():
        
        try:
            
            c = currentColors[int(row['peakOrd'])]
        
        except ValueError:
            
            c = 'black'
        
        if row['secondInd'] == 1:
            
            ls = '--'
        
        else:
            
            ls = '-.'
            
        ax2.axvline(x = row['time'], linestyle = ls, color = c)
    
    fig.tight_layout()
    
    if savePlots:
        
        fig.savefig(os.path.join(savePlots, namePrefix + str(df2.columns[0]) + nameSuffix), dpi = 300)
    
    if showPlots:
        
        plt.show()
        
    plt.close(fig)


def plotArchive(df, ind = None, savePlots = None, namePlot = 'archiveChanges', colorMap = 'rainbow'):
    
    """
    Visualize data revisions.
    
    Parameters
    -----
    df: pandas.DataFrame
        monthly pandas DataFrame, downloaded from MEI_ARCHIVE or similar
        (each row is one month, each column is one edition of the same
        variable), the last six characters of each column name need to be 
        in '%Y%m' format (e.g., 200105)
    ind: pandas.DataFrame
        pandas DataFrame (with one column, which is used to indicate vertical lines)
    savePlots: str or None
        path where to save plot
    namePlot: str
        plot name used when savePlots != None
    colorMap: str
        colormap code, see https://matplotlib.org/users/colormaps.html
        for examples
    """
    
    plotMe = df.copy()
    plotMe.dropna(how = 'all', inplace = True)
    
    
    # Labels and values
    
    plotTime = pd.DataFrame(data = list(range(len(plotMe))), index = plotMe.index, columns = ['time'])
    
    if (len(plotMe) < 12*25): 
        
        data_values = [row['time'] for index, row in plotTime.iterrows() if index.month == 1]
        data_labels = [index.strftime('%Y-%m') for index, row in plotTime.iterrows() if index.month == 1]
    
    else:
        
        data_values = [row['time'] for index, row in plotTime.iterrows() if (index.month == 1 & (index.year % 5 == 0))]
        data_labels = [index.strftime('%Y-%m') for index, row in plotTime.iterrows() if (index.month == 1 & (index.year % 5 == 0))]
    
                       
    # Color maps
    
    cmap = mpl.cm.get_cmap(name = colorMap)
    numCols = plotMe.shape[1]
    normalize = mpl.colors.Normalize(vmin = 0, vmax = numCols)
    
    firstYear = int(plotMe.columns[0][-6:-2])
    lastYear = int(plotMe.columns[-1][-6:-2])
    normalizeYears = mpl.colors.Normalize(vmin = firstYear, vmax = lastYear)
    
    
    # Plot
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 2.5))
    plt.xticks(data_values, data_labels, rotation = 60)
    
    for i, thisCol in enumerate(plotMe.columns):
    
        thisColor = cmap(normalize(i))
    
        ax.plot(plotTime[-plotMe[thisCol].isnull()]['time'], plotMe[-plotMe[thisCol].isnull()][thisCol], linestyle = '-', linewidth = 0.5, color = thisColor)
    
    scalarMappable = mpl.cm.ScalarMappable(norm = normalizeYears, cmap = cmap)
    scalarMappable.set_array(numCols)
    plt.colorbar(scalarMappable, orientation = 'vertical')
    
    
    # Vertical lines for index
    
    if isinstance(ind, pd.DataFrame):
        
        indExt = pd.concat([plotTime
                            , ind.rename(columns = {ind.columns[0]: 'ind'})]
                            , axis = 1)
        
        indExt = indExt[indExt['time'].notnull() & indExt['ind'].notnull() & (indExt['ind'] != 0)]
        
        if (len(indExt) > 0):
        
            for i, row in indExt.iterrows():
                # i, row = list(indExt.iterrows())[0]
            
                if row['ind'] == 1:
                    
                    ls = '--'
                
                elif row['ind'] == -1:
                    
                    ls = '-.'
                    
                ax.axvline(x = row['time'], linestyle = ls, color = 'gray')
    
    
    if savePlots:
        
        fig.savefig(os.path.join(savePlots, namePlot), dpi = 300, bbox_inches='tight')
    
    plt.show()


#def widgetThumbnail(f, thumbName, width = 1024, height = 768):
#    
#    """
#    Create .png file from .html file using PhantomJS.
#    """
#    
#    success = False
#    
#    try:
#        
#        phantom = os.environ['PHANTOMJS']
#        
#        print('Phantom path: %s' %(phantom))
#        
#    except KeyError:
#        
#        print('Error: PhantomJS dependency could not be found. Thumbnail could not be generated.')
#        
#        return(None)
#        
#    try:
#        
#        fjs = str(thumbName) + '.js'
#        
#        js = ('var page = require(\'webpage\').create();\n'
#            + 'page.viewportSize = { width: ' +  str(width) + ', height: ' + str(height) + ' };\n'
#            + 'page.clipRect = { top: 0, left: 0, width: ' +  str(width) + ', height: ' + str(height) + ' };\n'
#            + 'page.open(\'' + str(f) + '\', function(status) {\n'
#            + 'console.log("Status: " + status);\n'
#            + 'if(status === "success") {\n'
#            + 'page.render(\'' + str(thumbName) + '.png\');\n'
#            + '}\n'
#            + 'phantom.exit();\n'
#            + '});')
#        
#        file = open(fjs, 'w')
#        file.write(js)
#        file.close()
#        
#        res = call('phantomjs ' + str(fjs))
#        
#        if res != 'try-error':
#            
#            success = True
#            
#        if not(Path(fjs).is_file()):
#            
#            success = False
#            
#        if success:
#            
#            print('Thumbnail succesfully created.')
#            
#        else:
#            
#            print('Error: Could not create htmlwidget thumbnail. Empty thumbnail created.')
#            
#    except:
#        
#        print('Error: Could not open .html or create thumbnail.')