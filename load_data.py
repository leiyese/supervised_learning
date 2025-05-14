import pandas as pd
import numpy as np


def csv_to_df(filename):
    df = pd.read_csv(filename)
    return df
