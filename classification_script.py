from read_xls_file import *
from clf_models import *

from sklearn import preprocessing

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------- #

FILE_PATH = "data/iris.xls"
SHEET_NAME = "Iris"
df = read_xls_file(FILE_PATH, SHEET_NAME)
df.fillna(-99999, inplace=True)
df.dropna(inplace=True)

# print(df.head())

df_InputArray = np.array(df.drop(['iris'], 1))
df_InputArray = preprocessing.scale(df_InputArray)
df_OutputArray = np.array(df['iris'])
df_OutputArray, levels = pd.factorize(df_OutputArray)

# print(len(df_InputArray), len(df_OutputArray))

# ---------------------------------------------------------------------------------------------------- #

clf_svm_SVC_model(df_InputArray, df_OutputArray)
clf_dec_tree_model(df_InputArray, df_OutputArray)
clf_K_nearest_neighbour_model(df_InputArray, df_OutputArray, neighbors=3)
clf_K_nearest_neighbour_model(df_InputArray, df_OutputArray, neighbors=5)
clf_K_nearest_neighbour_model(df_InputArray, df_OutputArray, neighbors=7)
