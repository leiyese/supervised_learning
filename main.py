# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
from load_data import csv_to_df

df_test = csv_to_df("home-data-for-ml-course/test.csv")
df_train = csv_to_df("home-data-for-ml-course/train.csv")

df_test.head(), df_train.head()
# %%
