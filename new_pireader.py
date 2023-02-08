"""
General wrapper to interface with pi lime01 data server.

This code is designed to be able to be removed and used by
other projects to interact with the PIconnect library.

"""
import datetime
import logging
from typing import List

import numpy as np
import pandas as pd
from . import PIconnect as PI
from .PIconnect.PIConsts import CalculationBasis, SummaryType, TimestampCalculation
from datapipelines.pidata.PIconnect.AFSDK import AF

from ukpnutils.loggingdecorator import log, get_logger
from ukpnutils import tools

from datapipelines.pidata.PIconnect.time import timestamp_to_index, to_af_time_range, to_af_time


logger = get_logger()

# used to set default timestamps when not provided
DEFAULT_YEARS_OF_DATA = 1


def convert_datetime_to_datetime(dot_net_list):
    actual_dt_list = []
    for timestamps in dot_net_list:
        actual_dt_list.append(timestamp_to_index(timestamps))
    return actual_dt_list


# Numpy Datetime convert, here for later
def convert_datetime_to_datetime_numpy(dot_net_list):
    timestamps = np.array([dot_net_dt.ToFileTime() for dot_net_dt in dot_net_list], dtype='datetime64[us]')
    actual_dt_list = np.array(timestamps, dtype='datetime64[s]')
    actual_dt_list = actual_dt_list - np.datetime64('1970-01-01T00:00:00Z')
    return actual_dt_list.tolist()


class PIDataReader:
    """The main class for interfacing with PIconnect."""

    def __init__(self,
                 t1: datetime.datetime = None,
                 t2: datetime.datetime = None,
                 time_interval: int = 30,
                 retrieval_method: str = 'time_average',
                 fill_na: bool = True):
        """
        Initialize the class.

        Parameters
        ----------
        t1 : datetime.datetime
            Start time of the extract. Must be timezone aware.

        t2 : datetime.datetime
            End time of the extract. Must be timezone aware.

        time_interval: int
            Resolution of the extract. Defaults is 30.

        retrieval_method : str
            The retrieval method

        fill_na: bool = True
            This kwarg enables the user to either request the data as is from
            PI, Nans and all, or to have them filled through two steps, (1) is
            to linearly interpolate all data, (2) is to back and forward fill
            to complete edge cases at start and end of the time series.

        Returns
        -------
        None
        """
        # checks
        tools.check_type(time_interval, "time_interval", int)
        tools.check_type(retrieval_method, "retrieval_method", str)
        tools.check_type(fill_na, "fill_na", bool)

        # unpack args, kwargs and constants
        self.permissible_retrieval_methods = ['raw', 'raw_dict', 'sampled', 'time_average', 'time_average_discrete', 'event_average', 'min', 'max', 'new_event_average']
        self.time_interval = time_interval
        self.retrieval_method = retrieval_method
        self.fill_na = fill_na

        # check whether t1 or t2 are blank
        if t1 is None:
            # default to now minus the default number of years.
            self.t1 = (tools.rounded_dt_to_prev_x_minute(self.time_interval) - datetime.timedelta(days=DEFAULT_YEARS_OF_DATA * 365))
            self.t1.replace(tzinfo=datetime.timezone.utc)

        else:
            self.t1 = t1

        if t2 is None:
            # default to now, rounded to previous X minute. This is because we
            # must return an ordered time series according to the forecast
            # needs.
            self.t2 = tools.rounded_dt_to_prev_x_minute(self.time_interval)
            self.t2.replace(tzinfo=datetime.timezone.utc)
        else:
            self.t2 = t2

        return None

    @staticmethod
    def test_connection(server: str):
        """
        Test the connection to the PI AF Database.

        Parameters
        ----------
        server: str
            data, asset

        Returns
        -------
            Boolean,
                True if connection was established, False if not.

        Example
        -------
        from opforecast import pireader
        pr = pireader()
        pr.test_connection("data")
        """
        # check args
        tools.check_type(server, "server", str)
        if server not in ["data", "asset"]:
            raise ValueError(f"Incorrect server requested. Received {server} but should be either 'data or asset'")

        # try connect to either the data or asset server
        try:
            if server == 'data':
                # set the name
                expected_name = "Data_server_name_censored"
                # try connect to the server
                database = PI.PIServer()
            elif server == 'asset':
                # set the name
                expected_name = "AF_server_name_censored"
                database = PI.PIAFDatabase()

        # !!! find the actual error when failed connection.
        except TypeError:
            return False
        except BaseException:
            raise

        # if we have connected to LIME01, report it
        if database.server_name == expected_name:
            logger.debug(f"Connected to {expected_name} successfully.")
            return True

    @log
    def load_pitag_data(self, pitags: List[str]) -> pd.core.frame.DataFrame:
        """
        Load data from a given pitag from PI data.

        Parameters
        ----------
        pitags: List(str)
            list of pitag strings that you wish to retrieve.

        Inherited Parameters from self
        ------------------------------
        t1: datetime.datetime
            start time for data retrieval.
        t2: datetime.datetime
            end time for data retrieval.
        time_interval: int
            pi measurement interval, default 30. Must be in minutes
        retrieval_method: str
            method of retrieving pi data, options are: 'raw','raw_dict', 'sampled', 'time_average',
            'time_average_discrete', 'event_average', 'min', 'max','new_event_average'
        fill_na: bool = True
            This kwarg enables the user to either request the data as is from
            PI, Nans and all, or to have them filled through two steps, (1) is
            to linearly interpolate all data, (2) is to back and forward fill
            to complete edge cases at start and end of the time series.

        Returns
        -------
        pi_data: pandas.DataFrame OR pi_data: list
            pandas dataframe containing the retrieved pi data OR
            list of dataframes of raw data

        Example
        -------
        import datetime
        from opforecast import pireader
        with tools.Timer():
            pi_tags = ['E_0C8085_911-CCT_P','E_0C8085_931-CCT_P','E_0C8085_913-CCT_P','E_0C8085_933-CCT_P']
            t1 = datetime.datetime(2021,1,1,0,0,0,tzinfo=datetime.timezone.utc)
            t2 = datetime.datetime(2022,1,1,0,0,0,tzinfo=datetime.timezone.utc)
            pi_data = pireader.PIDataReader( t1 = t1, t2 = t2)
            data = pi_data.load_pitag_data(pi_tags)

        Thoughts
        --------
        The point of this function is to retrieve data from pi using a wrapper
        to the PIconnect module. In future we should consider rewriting the
        PIconnect module completely. This function extracts the timeseries for
        particular pitag that can be requested for a specfic time range. If no
        timerange then the last 2 years will be loaded.

        Could have a try number input parameter.

        Warning
        -------
        A lot of functions rely on the fact that there IS data. we must return
        null values if there is no data.

        time_interval default is 30
        """
        # check the inputs
        tools.check_type(pitags, "pitags", list)
        # assert that all input times are tz aware
        if self.t1.tzinfo is None or self.t1.tzinfo.utcoffset(self.t1) is None:
            raise ValueError("The input variable t1 is not timezone aware.")
        if self.t2.tzinfo is None or self.t2.tzinfo.utcoffset(self.t2) is None:
            raise ValueError("The input variable t2 is not timezone aware.")
        # check for incorrect retrieval_method
        if self.retrieval_method not in self.permissible_retrieval_methods:
            logger.error(f'Unrecognised pi retrieval method {self.retrieval_method}, expected {self.permissible_retrival_methods}.')
            raise ValueError(self.retrieval_method)

        # pre-allocate the retrieval
        responses = []

        # loop through each pitag to get the data
        server = PI.PIServer()
        try:
            data = self._retrieve_and_clean_data(pitags, server)
        except BaseException:
            try:
                data = self._retrieve_and_clean_data(pitags, server)
            except BaseException:
                Warning(f'Failed twice at getting data for list of pitags')
                logger.warning(f'Failed twice at getting data for list of pitags')
                data = pd.DataFrame()

        # responses is a list of list of lists. we need to remove the middle dim
        # responses_reshaped = [x[0] for x in responses]
        # # convert responses to a dataframe
        # df = pd.DataFrame(data=np.array(responses_reshaped).T, index=timestamps[0], columns=pitags)
        # Convert timestamps to UTC
        if self.retrieval_method == 'raw_dict':
            return data
        else:
            data = data.tz_convert('UTC')
            # replace pi string errors with nan
            data = data.apply(pd.to_numeric, errors='coerce')
            # if user requested nan's to be filled, fill them linearly then ff/bfill
            if self.fill_na:
                data.interpolate(method='linear', inplace=True)
                data = data.fillna(method="ffill", axis=0).fillna(method="bfill", axis=0)
            return data

    def _get_data(self, pi_tag):
        if self.retrieval_method == 'raw':
            return self._get_raw_data(pi_tag)
        elif self.retrieval_method == 'raw_dict':
            return self._get_raw_dict_data(pi_tag)
        elif self.retrieval_method == 'sampled':
            return self._get_sampled_data(pi_tag)
        elif self.retrieval_method == 'time_average':
            return self._get_time_average_data(pi_tag)
        elif self.retrieval_method == 'time_average_discrete':
            return self._get_time_average_discrete_data(pi_tag)
        elif self.retrieval_method == 'event_average':
            return self._get_event_average_data(pi_tag)
        elif self.retrieval_method == 'min':
            return self._get_min_data(pi_tag)
        elif self.retrieval_method == 'max':
            return self._get_max_data(pi_tag)
        elif self.retrieval_method == 'new_event_average':
            return self._get_new_event_average_data(pi_tag)

    # pi data functions we need
    # time weighted = 0
    # event weighted = 1

    @log
    def _get_sampled_data(self, pi_tag):

        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect
        time_interval = self.time_interval
        t1 = self.t1
        t2 = self.t2

        # Covert the Start time and end time to AF Time Range, and convert time_interval to AF Native TimeSpan, forcing UTC Timezone
        time_range = to_af_time_range(t1, t2)
        interval = AF.Time.AFTimeSpan.Parse(f'{time_interval}m', AF.Time.AFTimeZone.UtcTimeZone)

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.InterpolatedValues(time_range, interval, "", False, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}
        timestamp_pi = all_pipoint_sums[0]
        timestamp_array = timestamp_pi.GetValueArrays(list(), list(), list())  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        timestamp_list = convert_datetime_to_datetime(timestamp_array[1])  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        df_dict.update({"timestamp": timestamp_list})  # Loads only one timestamp array as its the same for all

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            actual_data_array = pipoint.GetValueArrays(list(), list(), list())
            df_dict.update({f"{pipoint.PIPoint.Name}": actual_data_array[0]})

        # Load Dict to DF and perform Transforms
        final_df = pd.DataFrame(df_dict)
        # final_df["timestamp"] = final_df["timestamp"].apply(lambda _: datetime.datetime.strptime(_, "%d/%m/%Y %H:%M:%S"))  # Don't ask why this was used, it was the only thing that worked, Don't touch until a Robust Solution is found
        final_df.set_index("timestamp", inplace=True, drop=True)
        return final_df

    @log
    def _get_raw_dict_data(self, pi_tag):

        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect, needs to be isotime with UTC forced which the parent function already does
        t1 = self.t1
        t2 = self.t2
        # Covert the Start time and end time to AF Time Range
        time_range = to_af_time_range(t1, t2)

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.RecordedValues(time_range, AF.Data.AFBoundaryType.Inside, "", False, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            actual_data_array = pipoint.GetValueArrays(list(), list(), list())
            timestamp_list = convert_datetime_to_datetime(actual_data_array[1])
            df_dict.update({f"{pipoint.PIPoint.Name}": {"timestamp": timestamp_list, "value": actual_data_array[0]}})

        # Since this method just returns a dict then we just return it, way faster and easier to do transforms later on
        return df_dict

    @log
    def _get_raw_data(self, pi_tag):
        # Borderline Unstable
        logger.info("Very Intensive and full of errors")

        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect
        time_interval = self.time_interval
        t1 = self.t1
        t2 = self.t2

        # Covert the Start time and end time to AF Time Range, and convert time_interval to AF Native TimeSpan, forcing UTC Timezone
        time_range = to_af_time_range(t1, t2)
        interval = AF.Time.AFTimeSpan.Parse(f'{time_interval}m')  # Exists here for some reason, function falls apart without it

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.RecordedValues(time_range, AF.Data.AFBoundaryType.Inside, "", False, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}
        timestamp_pi = all_pipoint_sums[0]
        timestamp_array = timestamp_pi.GetValueArrays(list(), list(), list())  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        timestamp_list = convert_datetime_to_datetime(timestamp_array[1])  # Convert C# data to string and then back to python (Bad Hack, found no better way)
        df_dict.update({"timestamp": timestamp_list})  # Loads only one timestamp array as its the same for all

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            actual_data_array = pipoint.GetValueArrays(list(), list(), list())
            df_dict.update({f"{pipoint.PIPoint.Name}": actual_data_array[0]})

        # Load Dict to DF and perform Transforms
        final_df = pd.DataFrame.from_dict(df_dict, orient='index')
        # final_df["timestamp"] = final_df["timestamp"].apply(lambda _: datetime.datetime.strptime(_, "%d/%m/%Y %H:%M:%S"))
        final_df.set_index("timestamp", inplace=True, drop=True)
        return final_df

    @log
    def _get_time_average_data(self, pi_tag):

        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect
        time_interval = self.time_interval
        t1 = self.t1
        t2 = self.t2

        # Covert the Start time and end time to AF Time Range, and convert time_interval to AF Native TimeSpan, forcing UTC Timezone
        time_range = to_af_time_range(t1, t2)
        interval = AF.Time.AFTimeSpan.Parse(f'{time_interval}m', AF.Time.AFTimeZone.UtcTimeZone)

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.Summaries(time_range, interval, AF.Data.AFSummaryTypes.Average, AF.Data.AFCalculationBasis.TimeWeighted, AF.Data.AFTimestampCalculation.EarliestTime, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}
        timestamp_pi = all_pipoint_sums[0]
        timestamp_data = timestamp_pi.Values
        timestamp_data = list(timestamp_data)
        timestamp_data = timestamp_data[0]
        timestamp_array = timestamp_data.GetValueArrays(list(), list(), list())  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        timestamp_list = convert_datetime_to_datetime(timestamp_array[1])  # Convert C# data to string and then back to python (Bad Hack, found no better way)
        df_dict.update({"timestamp": timestamp_list})  # Loads only one timestamp array as its the same for all

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            pi_values = pipoint.Values
            actual_data = list(pi_values)
            actual_data = actual_data[0]
            actual_data_array = actual_data.GetValueArrays(list(), list(), list())
            df_dict.update({f"{actual_data.PIPoint.Name}": actual_data_array[0]})

         # Load Dict to DF and perform Transforms
        final_df = pd.DataFrame(df_dict)
        # final_df["timestamp"] = final_df["timestamp"].apply(lambda _: datetime.datetime.strptime(_, "%d/%m/%Y %H:%M:%S"))  # Don't ask why this was used, it was the only thing that worked, Don't touch until a Robust Solution is found
        final_df.set_index("timestamp", inplace=True, drop=True)
        return final_df

    @log
    def _get_time_average_discrete_data(self, pi_tag):
        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect
        time_interval = self.time_interval
        t1 = self.t1
        t2 = self.t2

        # Covert the Start time and end time to AF Time Range, and convert time_interval to AF Native TimeSpan, forcing UTC Timezone
        time_range = to_af_time_range(t1, t2)
        interval = AF.Time.AFTimeSpan.Parse(f'{time_interval}m', AF.Time.AFTimeZone.UtcTimeZone)

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.Summaries(time_range, interval, AF.Data.AFSummaryTypes.Average, AF.Data.AFCalculationBasis.TimeWeightedDiscrete, AF.Data.AFTimestampCalculation.EarliestTime, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}
        timestamp_pi = all_pipoint_sums[0]
        timestamp_data = timestamp_pi.Values
        timestamp_data = list(timestamp_data)
        timestamp_data = timestamp_data[0]
        timestamp_array = timestamp_data.GetValueArrays(list(), list(), list())  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        timestamp_list = convert_datetime_to_datetime(timestamp_array[1])  # Convert C# data to string and then back to python (Bad Hack, found no better way)
        df_dict.update({"timestamp": timestamp_list})  # Loads only one timestamp array as its the same for all

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            pi_values = pipoint.Values
            actual_data = list(pi_values)
            actual_data = actual_data[0]
            actual_data_array = actual_data.GetValueArrays(list(), list(), list())
            df_dict.update({f"{actual_data.PIPoint.Name}": actual_data_array[0]})

         # Load Dict to DF and perform Transforms
        final_df = pd.DataFrame(df_dict)
        # final_df["timestamp"] = final_df["timestamp"].apply(lambda _: datetime.datetime.strptime(_, "%d/%m/%Y %H:%M:%S"))  # Don't ask why this was used, it was the only thing that worked, Don't touch until a Robust Solution is found
        final_df.set_index("timestamp", inplace=True, drop=True)
        return final_df

    @log
    def _get_event_average_data(self, pi_tag):

        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect
        time_interval = self.time_interval
        t1 = self.t1
        t2 = self.t2

        # Covert the Start time and end time to AF Time Range, and convert time_interval to AF Native TimeSpan, forcing UTC Timezone
        time_range = to_af_time_range(t1, t2)
        interval = AF.Time.AFTimeSpan.Parse(f'{time_interval}m', AF.Time.AFTimeZone.UtcTimeZone)

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.Summaries(time_range, interval, AF.Data.AFSummaryTypes.Average, AF.Data.AFCalculationBasis.EventWeighted, AF.Data.AFTimestampCalculation.EarliestTime, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}
        timestamp_pi = all_pipoint_sums[0]
        timestamp_data = timestamp_pi.Values
        timestamp_data = list(timestamp_data)
        timestamp_data = timestamp_data[0]
        timestamp_array = timestamp_data.GetValueArrays(list(), list(), list())  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        timestamp_list = convert_datetime_to_datetime(timestamp_array[1])  # Convert C# data to string and then back to python (Bad Hack, found no better way)
        df_dict.update({"timestamp": timestamp_list})  # Loads only one timestamp array as its the same for all

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            pi_values = pipoint.Values
            actual_data = list(pi_values)
            actual_data = actual_data[0]
            actual_data_array = actual_data.GetValueArrays(list(), list(), list())
            df_dict.update({f"{actual_data.PIPoint.Name}": actual_data_array[0]})

        # Load Dict to DF and perform Transforms
        final_df = pd.DataFrame(df_dict)
        # final_df["timestamp"] = final_df["timestamp"].apply(lambda _: datetime.datetime.strptime(_, "%d/%m/%Y %H:%M:%S"))  # Don't ask why this was used, it was the only thing that worked, Don't touch until a Robust Solution is found
        final_df.set_index("timestamp", inplace=True, drop=True)
        return final_df

    @log
    def _get_new_event_average_data(self, pi_tag):
        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect
        time_interval = self.time_interval
        t1 = self.t1
        t2 = self.t2

        # Covert the Start time and end time to AF Time Range, and convert time_interval to AF Native TimeSpan, forcing UTC Timezone
        time_range = to_af_time_range(t1, t2)
        interval = AF.Time.AFTimeSpan.Parse(f'{time_interval}m', AF.Time.AFTimeZone.UtcTimeZone)

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.Summaries(time_range, interval, AF.Data.AFSummaryTypes.Average, AF.Data.AFCalculationBasis.EventWeightedIncludeBothEnds, AF.Data.AFTimestampCalculation.Auto, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}
        timestamp_pi = all_pipoint_sums[0]
        timestamp_data = timestamp_pi.Values
        timestamp_data = list(timestamp_data)
        timestamp_data = timestamp_data[0]
        timestamp_array = timestamp_data.GetValueArrays(list(), list(), list())  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        timestamp_list = convert_datetime_to_datetime(timestamp_array[1])  # Convert C# data to string and then back to python (Bad Hack, found no better way)
        df_dict.update({"timestamp": timestamp_list})  # Loads only one timestamp array as its the same for all

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            pi_values = pipoint.Values
            actual_data = list(pi_values)
            actual_data = actual_data[0]
            actual_data_array = actual_data.GetValueArrays(list(), list(), list())
            df_dict.update({f"{actual_data.PIPoint.Name}": actual_data_array[0]})

        # Load Dict to DF and perform Transforms
        final_df = pd.DataFrame(df_dict)
        # final_df["timestamp"] = final_df["timestamp"].apply(lambda _: datetime.datetime.strptime(_, "%d/%m/%Y %H:%M:%S"))  # Don't ask why this was used, it was the only thing that worked, Don't touch until a Robust Solution is found
        final_df.set_index("timestamp", inplace=True, drop=True)
        return final_df

    @log
    def _get_min_data(self, pi_tag):

        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect
        time_interval = self.time_interval
        t1 = self.t1
        t2 = self.t2

        # Covert the Start time and end time to AF Time Range, and convert time_interval to AF Native TimeSpan, forcing UTC Timezone
        time_range = to_af_time_range(t1, t2)
        interval = AF.Time.AFTimeSpan.Parse(f'{time_interval}m', AF.Time.AFTimeZone.UtcTimeZone)

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.Summaries(time_range, interval, AF.Data.AFSummaryTypes.Minimum, AF.Data.AFCalculationBasis.TimeWeighted, AF.Data.AFTimestampCalculation.EarliestTime, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}
        timestamp_pi = all_pipoint_sums[0]
        timestamp_data = timestamp_pi.Values
        timestamp_data = list(timestamp_data)
        timestamp_data = timestamp_data[0]
        timestamp_array = timestamp_data.GetValueArrays(list(), list(), list())  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        timestamp_list = convert_datetime_to_datetime(timestamp_array[1])  # Convert C# data to string and then back to python (Bad Hack, found no better way)
        df_dict.update({"timestamp": timestamp_list})  # Loads only one timestamp array as its the same for all

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            pi_values = pipoint.Values
            actual_data = list(pi_values)
            actual_data = actual_data[0]
            actual_data_array = actual_data.GetValueArrays(list(), list(), list())
            df_dict.update({f"{actual_data.PIPoint.Name}": actual_data_array[0]})

        # Load Dict to DF and perform Transforms
        final_df = pd.DataFrame(df_dict)
        # final_df["timestamp"] = final_df["timestamp"].apply(lambda _: datetime.datetime.strptime(_, "%d/%m/%Y %H:%M:%S"))  # Don't ask why this was used, it was the only thing that worked, Don't touch until a Robust Solution is found
        final_df.set_index("timestamp", inplace=True, drop=True)
        return final_df

    @log
    def _get_max_data(self, pi_tag):

        # Load Time Interval and StartTime and End Time via a string for to_af_time_range function of PI Connect
        time_interval = self.time_interval
        t1 = self.t1
        t2 = self.t2

        # Covert the Start time and end time to AF Time Range, and convert time_interval to AF Native TimeSpan, forcing UTC Timezone
        time_range = to_af_time_range(t1, t2)
        interval = AF.Time.AFTimeSpan.Parse(f'{time_interval}m', AF.Time.AFTimeZone.UtcTimeZone)

        # Setup PIConfig, needed to parse large rows of data
        paging_config = AF.PI.PIPagingConfiguration(AF.PI.PIPageType.TagCount, 1000000, 5400, None, None)

        # Pull Values for all Pitags using PISDK Functions and arguements
        pipointlist = pi_tag.Summaries(time_range, interval, AF.Data.AFSummaryTypes.Maximum, AF.Data.AFCalculationBasis.TimeWeighted, AF.Data.AFTimestampCalculation.EarliestTime, paging_config)

        # Transform the values into a tuple for faster parse, and since the object returned by the function is non iteratable
        all_pipoint_sums = tuple(pipointlist)
        df_dict = {}
        timestamp_pi = all_pipoint_sums[0]
        timestamp_data = timestamp_pi.Values
        timestamp_data = list(timestamp_data)
        timestamp_data = timestamp_data[0]
        timestamp_array = timestamp_data.GetValueArrays(list(), list(), list())  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        timestamp_list = convert_datetime_to_datetime(timestamp_array[1])  # Pull All values with timestamps into arrays(PI Native Method, Don't ask why)
        df_dict.update({"timestamp": timestamp_list})  # Loads only one timestamp array as its the same for all

        # Loop Through all the PIPoints and get their values and add to dict
        for pipoint in all_pipoint_sums:
            pi_values = pipoint.Values
            actual_data = list(pi_values)
            actual_data = actual_data[0]
            actual_data_array = actual_data.GetValueArrays(list(), list(), list())
            df_dict.update({f"{actual_data.PIPoint.Name}": actual_data_array[0]})

        # Load Dict to DF and perform Transforms
        final_df = pd.DataFrame(df_dict)
        # final_df["timestamp"] = final_df["timestamp"].apply(lambda _: datetime.datetime.strptime(_, "%d/%m/%Y %H:%M:%S"))  # Don't ask why this was used, it was the only thing that worked, Don't touch until a Robust Solution is found
        final_df.set_index("timestamp", inplace=True, drop=True)
        return final_df

    @log
    def _retrieve_and_clean_data(self, points, server):
        """Get the data and clean it."""
        # logger.info(f'Getting data for {point}')
        # Get data from pi server as per retrieval method
        return self._get_data(server.search_list(points))
