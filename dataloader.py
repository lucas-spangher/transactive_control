import pandas as pd
import pytz

def convert_times(df):
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(pytz.timezone('Singapore'),ambiguous='infer')
    df['TimeStamp'] = df['TimeStamp'].dt.tz_convert(pytz.timezone('America/Los_Angeles'))
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp']) - pd.Timedelta(1, unit='h')
    return df
