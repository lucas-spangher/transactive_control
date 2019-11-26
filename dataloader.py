import pandas as pd
import pytz

def convert_times(df):
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    df['TimeStamp'] = df['TimeStamp'].dt.tz_localize(pytz.timezone('Singapore'),ambiguous='infer')
    df['TimeStamp (ST)'] = df['TimeStamp'].dt.tz_convert(pytz.timezone('America/Los_Angeles'))
    df['TimeStamp (ST)'] = pd.to_datetime(df['TimeStamp (ST)']) - pd.Timedelta(1, unit='h')
    df['Day (ST)'] = df['TimeStamp (ST)'].dt.date
    df['Hour (ST)'] = df['TimeStamp (ST)'].dt.hour
    df = df.sort_values(by=['Day (ST)', 'Hour (ST)'])

    return df
