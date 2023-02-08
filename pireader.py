"""
General wrapper to interface with pi lime01 data server.

This code is designed to be able to be removed and used by
other projects to interact with the PIconnect library.

Last updated by Harshil Sumaria on 10th August 2022
"""
import datetime
import logging
from typing import List

import numpy as np
import pandas as pd
from . import PIconnect as PI
from .PIconnect.PIConsts import CalculationBasis, SummaryType, TimestampCalculation

from ukpnutils.loggingdecorator import log, get_logger
from ukpnutils import tools

logger = get_logger()

# used to set default timestamps when not provided
DEFAULT_YEARS_OF_DATA = 2


class PIDataReader:
    """The main class for interfacing with PIconnect."""

    def __init__(self,
                 t1: datetime.datetime = None,
                 t2: datetime.datetime = None,
                 time_interval: int = 30,
                 retrieval_method: str = 'time_average'):
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

        Returns
        -------
        None
        """
        # checks
        tools.check_type(time_interval, "time_interval", int)
        tools.check_type(retrieval_method, "retrieval_method", str)

        # unpack args, kwargs and constants
        self .permissible_retrieval_methods = ['raw', 'sampled', 'time_average', 'event_average', 'min', 'max']
        self.time_interval = time_interval
        self.retrieval_method = retrieval_method

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
            logger.info(f"Connected to {expected_name} successfully.")
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
            method of retrieving pi data, options are: 'raw','sampled',
            'time_average','event_average','min','max'.

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
        with PI.PIServer() as server:
            for point in pitags:
                # try except block to try get the data twice in case pi says no
                # retrieves data from pi and removes non-numeric values e.g. Bad Data
                # and joins it onto res
                try:
                    data, timestamps = self._retrieve_and_clean_data(point, server)
                    responses.append(data)
                except BaseException:
                    try:
                        data, timestamps = self._retrieve_and_clean_data(point, server)
                        responses.append(data)
                    except BaseException:
                        Warning(f'Failed twice at getting data for {point}')
                        logger.warning(f'Failed twice at getting data for {point}')
                        pass

        # responses is a list of list of lists. we need to remove the middle dim
        responses_reshaped = [x[0] for x in responses]
        # convert responses to a dataframe
        df = pd.DataFrame(data=np.array(responses_reshaped).T, index=timestamps[0], columns=pitags)
        # Convert timestamps to UTC
        df = df.tz_convert('UTC')
        # replace pi string errors with nan
        df = df.apply(pd.to_numeric, errors='coerce')
        df.interpolate(method='linear', inplace=True)
        df = df.fillna(method="ffill", axis=0).fillna(method="bfill", axis=0)
        return df

    def _get_data(self, pi_tag):
        if self.retrieval_method == 'raw':
            return self._get_raw_data(pi_tag)
        elif self.retrieval_method == 'sampled':
            return self._get_sampled_data(pi_tag)
        elif self.retrieval_method == 'time_average':
            return self._get_event_average_data(pi_tag)
        elif self.retrieval_method == 'event_average':
            return self._get_event_average_data(pi_tag)
        elif self.retrieval_method == 'min':
            return self._get_min_data(pi_tag)
        elif self.retrieval_method == 'max':
            return self._get_max_data(pi_tag)

    # pi data functions we need
    # time weighted = 0
    # event weighted = 1
    @log
    def _get_sampled_data(self, pi_tag):
        return pi_tag.interpolated_values(self.t1, self.t2, f'{self.time_interval}m')

    @log
    def _get_raw_data(self, pi_tag):
        return pi_tag.recorded_values(self.t1, self.t2)

    @log
    def _get_time_average_data(self, pi_tag):
        return pi_tag.summaries(start_time=self.t1, end_time=self.t2, interval=f'{self.time_interval}m', summary_types=SummaryType.AVERAGE, calculation_basis=CalculationBasis.TIME_WEIGHTED, time_type=TimestampCalculation.EARLIEST_TIME)

    @log
    def _get_event_average_data(self, pi_tag):
        return pi_tag.summaries(start_time=self.t1, end_time=self.t2, interval=f'{self.time_interval}m', summary_types=SummaryType.AVERAGE, calculation_basis=CalculationBasis.EVENT_WEIGHTED, time_type=TimestampCalculation.EARLIEST_TIME)

    @log
    def _get_min_data(self, pi_tag):
        return pi_tag.summaries(start_time=self.t1, end_time=self.t2, interval=f'{self.time_interval}m', summary_types=SummaryType.MINIMUM, time_type=TimestampCalculation.EARLIEST_TIME)

    @log
    def _get_max_data(self, pi_tag):
        return pi_tag.summaries(start_time=self.t1, end_time=self.t2, interval=f'{self.time_interval}m', summary_types=SummaryType.MAXIMUM, time_type=TimestampCalculation.EARLIEST_TIME)

    @log
    def _retrieve_and_clean_data(self, point, server):
        """Get the data and clean it."""
        # print(f'Getting data for {point}')
        # Get data from pi server as per retrieval method
        return self._get_data(server.search(point)[0])

