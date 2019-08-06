import numpy as np
import pandas as pd

import glob
from natsort import natsorted

from astropy.time import Time
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import lightkurve as lk

import warnings

import seaborn as sns
sns.set_context('talk')

from scipy.interpolate import splprep, splev


campaign_dict = {'C0':0,'C1':1,'C2':2,'C3':3,'C4':4,'C5':5,'C6':6,'C7':7,'C8':8,
                 'C9a':91,'C9b':92,'C10b':101,'C11a':111,'C11b':112,
                 'C12':12,'C13':13,'C14':14,'C15':15,'C16':16,'C17':17,
                 'C18':18,'C19':19}


def get_temperature_data_from_suffix(file_suffix='BoardTemperatures'):
    """Get all-campaign temperature telemetry given a filetype

    Parameters
    ----------
    file_suffix : string
        Must be one of
            - 'BoardTemperatures' (default)
            - 'TelescopeTemperatureTH_2',
            - 'TelescopeTemperatureTH_1'
            - 'TelescopeTemperaturePED'

    Returns
    -------
    df : pandas DataFrame
        A DataFrame containing all temperature telemetry for all campaigns

    """
    txt_files = glob.glob('../data/Aed/K2/thermal/*.txt')
    txt_files = natsorted(txt_files)

    df_all_campaigns = pd.DataFrame()
    for campaign in campaign_dict.keys():
        fn = '../data/AED/K2/thermal/{}_{}.txt'.format(campaign, file_suffix)
        df = pd.read_csv(fn, skiprows=[0,1,2,3,4,6], sep='|')
        df['campaign']=campaign_dict[campaign]
        df_all_campaigns = df_all_campaigns.append(df, ignore_index=True)

    return df_all_campaigns



def pre_process_temperature_data(df, add_jitter=False):
    """Pre-process the data in preparation for use as regressors.

        Peforms these tasks:
            - Convert to Kelvin from Celsius
            - Make index into DateTimeIndex for later date matching
            - (optionally) Add Gaussian noise comparable to discretization


    Parameters
    ----------
    df : pandas DataFrame
        A DataFrame containing all raw temperature telemetry for all campaigns

    Returns
    -------
    df : pandas DataFrame
        Reformatted DataFrame containing processed temperature telemetry

    """

    time_vector = Time(df.MJD.values, format='mjd', scale='tdb')
    df = df.set_index(pd.DatetimeIndex(time_vector.datetime))

    temp_cols = df.columns[3:-1]
    for col in temp_cols:
        df[col] = 273.15 + df[col]
        if add_jitter:
            scale = 0.011 # experimentally determined discretization noise
            df[col] += np.random.normal(loc=0, scale=0.011, size=len(df))

    return df
