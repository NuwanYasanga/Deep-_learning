import pandas as pd
import numpy as np
import os
import math
import random
from tqdm.notebook import tqdm
from operator import itemgetter
from functools import reduce
from statistics import mean
from statistics import stdev

user_1 = 'C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_FT_100_latest/User_1@sample_1.csv'

data_frame_desktop_bf1 = pd.read_csv(user_1, header=0)
data_frame_desktop_bf2 = data_frame_desktop_bf1[(data_frame_desktop_bf1['F1']>(-5000)) & 
                                                   (data_frame_desktop_bf1['F2']>(-5000))]

