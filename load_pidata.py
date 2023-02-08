"""
PI Data loading module.

This module manages all the data management features of the operational
forecasting software. All loading, saving, quality control, regularized
manipulation, etc., that are repeatable and common should exist in this
module.

To Do
----
Asset framework:
    Need a method of extracting pitags more intelligently.

Data quality:
    Need an improved data quality approach.
"""
import datetime
import logging
from os import PathLike
from typing import List

import numpy as np
import pandas as pd

from ukpnutils.tools import check_file, check_type
from ukpnutils.loggingdecorator import log, get_logger
from datapipelines.pidata import pireader, new_pireader


logger = get_logger()

# default file name for test/dummy data
TEST_DATA = ""


@log
def load(pitags: List[str],
         t1: datetime.datetime,
         t2: datetime.datetime,
         intervals: int = 30,
         retrieval_method: str = 'time_average',
         fill_na: bool = True) -> pd.core.frame.DataFrame:
    """
    Load pitag data from our available sources using a wrapper.

    This function returns the output from the appropriate function depending if
    we had connection to PI, or whether we use a local file or database.

    Parameters
    ----------
    pitags : [str]
        List of pitag names.
    t1 : datetime.datetime, tz aware.
        Start time, optional.
    t2 : datetime.datetime, tz aware.
        End time, optional.
    intervals : int
        The time intervals between timestamps.
    retrieval_method : str
        The type of retrival method if extracting from pi. The acceptable
        options are ['raw','raw_dict', 'sampled', 'time_average','time_average_discrete', 
        'event_average', 'min', 'max','new_event_average']
    fill_na: bool = True
        This kwarg enables the user to either request the data as is from
        PI, Nans and all, or to have them filled through two steps, (1) is
        to linearly interpolate all data, (2) is to back and forward fill
        to complete edge cases at start and end of the time series.

    Returns
    -------
    data : pd.core.frame.DataFrame, unless 'raw" is the retrieval_mehod, in
        which case a list of dataframes is returned.
    """
    # checks
    check_type(pitags, "pitags", list)
    for pitag in pitags:
        check_type(pitag, "pitag in pitags", str)
    check_type(t1, "t1", datetime.datetime)
    check_type(t2, "t2", datetime.datetime)
    check_type(intervals, "intervals", int)
    check_type(retrieval_method, "retrieval_method", str)
    permissible_methods = ['raw', 'raw_dict', 'sampled', 'time_average', 'time_average_discrete', 'event_average', 'min', 'max', 'new_event_average']
    if retrieval_method not in permissible_methods:
        raise ValueError(
            f"Variable 'retrieval_method' is not recognised. Received {retrieval_method}. Permissible options are {permissible_methods}")
    check_type(fill_na, "fill_na", int)

    # run the method
    if new_pireader.PIDataReader.test_connection("data"):
        data = load_pitag_data_from_pi(pitags, t1, t2, intervals, retrieval_method, fill_na=fill_na)
    else:
        data = load_pitag_data_from_file(pitags, t1, t2, intervals, fname=TEST_DATA, fill_na=fill_na)
    return data


@log
def load_pitag_data_from_pi(pitags: List[str],
                            t1: datetime.datetime,
                            t2: datetime.datetime,
                            intervals: int = 30,
                            retrieval_method: str = 'time_average',
                            fill_na: bool = True) -> pd.core.frame.DataFrame:
    """
    Interface with pireader class and get the required data.

    Parameters
    ----------
    pitags : [str]
        List of pitag names.
    t1 : datetime.datetime, tz aware.
        Start time, optional.
    t2 : datetime.datetime, tz aware.
        End time, optional.
    intervals : int
        The time intervals between timestamps.
    retrieval_method : str
        The type of retrival method if extracting from pi. The acceptable
        options are ['raw', 'sampled', 'time_average', 'event_average', 'min',
                     'max']
    fill_na: bool = True
        This kwarg enables the user to either request the data as is from
        PI, Nans and all, or to have them filled through two steps, (1) is
        to linearly interpolate all data, (2) is to back and forward fill
        to complete edge cases at start and end of the time series.

    Returns
    -------
    pi_data : pd.core.frame.DataFrame, unless 'raw" is the retrieval_mehod, in
        which case a list of dataframes is returned.
    """
    # checks
    check_type(pitags, "pitags", list)
    for pitag in pitags:
        check_type(pitag, "pitag in pitags", str)
    check_type(t1, "t1", datetime.datetime)
    check_type(t2, "t2", datetime.datetime)
    # assume that all input times are tz aware
    if t1.tzinfo is None or t1.tzinfo.utcoffset(t1) is None:
        raise TypeError("The input variable t1 is not timezone aware.")
    if t2.tzinfo is None or t2.tzinfo.utcoffset(t2) is None:
        raise TypeError("The input variable t2 is not timezone aware.")
    check_type(intervals, "intervals", int)
    check_type(retrieval_method, "retrieval_method", str)
    permissible_methods = ['raw', 'raw_dict', 'sampled', 'time_average', 'time_average_discrete', 'event_average', 'min', 'max', 'new_event_average']
    if retrieval_method not in permissible_methods:
        raise ValueError(
            f"Variable 'retrieval_method' is not recognised. Received {retrieval_method}. Permissible options are {permissible_methods}")
    check_type(fill_na, "fill_na", int)

    # get an instance of the PIDataReader Class and extract the data.
    reader = new_pireader.PIDataReader(retrieval_method=retrieval_method, t1=t1, t2=t2, time_interval=intervals, fill_na=fill_na)
    pi_data = reader.load_pitag_data(pitags)
    # raise exception if empty?
    if not retrieval_method == 'raw_dict':
        if pi_data.empty:
            raise ValueError("load_pitag_data_from_pi: pi_data from PI is empty")
    return pi_data


@log
def load_pitag_data_from_file(pitags: List[str],
                              t1: datetime.datetime,
                              t2: datetime.datetime,
                              intervals: int = 30,
                              fname: PathLike = TEST_DATA,
                              fill_na: bool = True) -> pd.core.frame.DataFrame:
    """
    Load all PI data.

    Parameters
    ----------
        pitags: [str]
            List of pitag names.
        t1: datetime.datetime
            Start time, optional.
        t2: datetime.datetime
            End time, optional.
        intervals : int,
            The time gap between measurements in minutes. Default = 30.
        fname : pathlike
            The full file path to the backup data file.Default = TEST_DATA
        fill_na: bool = True
            This kwarg enables the user to either request the data as is from
            PI, Nans and all, or to have them filled through two steps, (1) is
            to linearly interpolate all data, (2) is to back and forward fill
            to complete edge cases at start and end of the time series.

        If t1 and t2 are None, then the entire history is loaded.

    Returns
    -------
        data_qc : pd.core.frame.DataFrame.

    Thoughts
    --------
        This function extracts the timeseries for a particular pitag that can
        be requested for a specfic time range. All data will be loaded if not
        specifically asked for.

        In future this will become laoding from the soft backup of pi.
        Rename to load_from backup, check hostname and load from backup or
        files accordingly

    Warning
    -------
        A lot of functions rely on the fact that there IS data. we must return
        null values if there is no data.
    """
    # checks
    check_type(pitags, "pitags", list)
    for pitag in pitags:
        check_type(pitag, "pitag in pitags", str)
    check_type(t1, "t1", datetime.datetime)
    check_type(t2, "t2", datetime.datetime)
    # assume that all input times are tz aware
    if t1.tzinfo is None or t1.tzinfo.utcoffset(t1) is None:
        raise TypeError("The input variable t1 is not timezone aware.")
    if t2.tzinfo is None or t2.tzinfo.utcoffset(t2) is None:
        raise TypeError("The input variable t2 is not timezone aware.")
    check_type(intervals, "intervals", int)
    check_type(fill_na, "fill_na", int)

    # temporary load all from file.
    # !!!THIS MUST BE REPLACED WITH A ROBUST FUNCTION THAT TAKES PITAGS AND RETURNS A DATAFRAME OF ALL PITAGS REQUESTED FOR WHOLE HISTORY. Preferably we have a hot backup on the server.
    file_ok, (status, msg) = check_file(fname, ftype='.csv')
    if file_ok:
        data = pd.read_csv(fname, na_values=['[-11059] No Good Data For Calculation'], index_col='DATETIME')
        # need to convert timezone to UTC else we have non sequential datetimes when clocks go backwards
        data.set_index(pd.to_datetime(data.index).tz_convert('UTC'), inplace=True)
    else:
        raise Exception(f'Cannot open {fname} because check_file status {status}: {msg}')

    # we  may be in hindcast mode, if so, we cannot use future data to inform
    # the forecast, else a training/testing overlap.
    timestamps = pd.to_datetime(data.index)
    data_clipped = data[(timestamps >= t1) & (timestamps <= t2)]

    # isolate to just the requested pitags.
    data_isolated = np.full((data_clipped.values.shape[0], len(pitags)), np.nan)
    for i, pitag in enumerate(pitags):
        try:
            data_isolated[:, i] = data_clipped[pitag]
        except IndexError:
            pass
    data_isolated = pd.DataFrame(data_isolated, columns=pitags, index=data_clipped.index)

    # perform some quality control
    if data.shape[0] > 24 * 8:
        data_qc, _, _ = quality_control_pitag_data(data_isolated)
    else:
        data_qc = data_clipped.copy()

    if fill_na:
        # forward fill nans and resample the data to the same frequency as the forecast data
        # data_qc = data_qc.fillna(method='ffill')
        data_qc = data_qc.resample(f'{intervals}min').mean().interpolate()

    return data_qc


@log
def quality_control_pitag_data(data: pd.core.frame.DataFrame):
    """
    Quality control the raw PI data.

    Parameters
    ----------
    data: pd.core.frame.DataFrame
        PI tag data time series for QC measures, measures are automatically
        detected from the type of variable(e.g., power / voltage etc).

    Returns
    -------
    data_qc: pd.core.frame.DataFrame
        Same data as input, after quality control.
    frac_removed: np.float64
        The fraction of data removed during quality control
    var: list of str
        The quantity variable extracted from the PI tag

    Thoughts
    --------
    The intention here is that any data from PI get subjected to quality
    control measures. We must devise algorithms to automatically null any
    suspected spurious data. We must have specific QC per type of variable
    being subject to QC.
    """
    # find what variable we are dealing with
    var = [pitag.split('_')[-1].split('/')[-1] for pitag in data.columns]
    # option to filter using var for power/voltage/current specific filtering

    # take a smooth median from a 1-week moving window, and interpolate for nan
    filt_data = data.rolling('7D').median().interpolate()
    # build a check for how far a measurement extends from the smoothed series
    check = data - filt_data
    # derive upper and lower thresholds using 1st and 99th percentiles
    upper_thres = check.dropna().quantile(0.99)
    lower_thres = check.dropna().quantile(0.01)
    # create a copy of the input
    data_qc = data.copy()
    # remove data points that violate the threshold checks
    inds = np.logical_or(check.values > upper_thres.values, check.values < lower_thres.values)
    data_qc[inds] = np.nan
    # report amount of removed data
    frac_removed = np.sum(inds) / data.shape[0]

    return data_qc, frac_removed, var
