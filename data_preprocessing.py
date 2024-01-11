import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

selected_features = ['Error_rate_%', 'WPM','neg_UD_%', 'mean_F1', 'mean_F2', 'mean_F3','mean_F4', 'Tri_graph', 'mean_hold_time',
       'mean_F1_dis_1_LL', 'mean_F1_dis_1_RR', 'mean_F1_dis_2_LL',
       'mean_F1_dis_2_RR', 'mean_F1_dis_3_LL', 'mean_F1_dis_3_RR',
       'mean_F2_dis_1_LL', 'mean_F2_dis_2_LL', 'mean_F2_dis_2_RR',
       'mean_F2_dis_3_LL', 'mean_F2_dis_3_RR', 'mean_F3_dis_2_RR',
       'mean_F3_dis_3_LL', 'mean_F4_dis_1_LL', 'mean_F4_dis_2_LL',
       'mean_F4_dis_2_RR', 'mean_F4_dis_3_LL', 'mean_F1_se', 'mean_F2_se',
       'mean_F3_se', 'mean_F4_se', 'mean_F1_th', 'mean_F2_th', 'mean_F3_th',
       'mean_F4_th', 'mean_F1_he']

def data_prep(df):

    #all_train_set  = pd.DataFrame()
    #all_test_set = pd.DataFrame()

    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []

    for a,i in enumerate (df['User'].unique()):
        user_df = df[df['User']==i]
        X = user_df[selected_features].values 
        y = user_df['User'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42) 

        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    #    train_set = df[df['User']==i].sample(frac=0.7, random_state=42)
    #    sample_ids = train_set['sample']
    #    test_set = df[(df['User']==i) & ~ df['sample'].isin (sample_ids) ]
    
    #    all_train_set = all_train_set.append(train_set,ignore_index=True)
    #    all_test_set = all_test_set.append(test_set, ignore_index=True)

    #X_train = pd.DataFrame(all_train_set.iloc[:,3:])
    #y_train =all_train_set.iloc[:,1]
    #X_test = pd.DataFrame(all_test_set.iloc[:,3:])
    #y_test = all_test_set.iloc[:,1]

    return X_train, y_train, X_test, y_test

#all_users = pd.read_csv('C:/Users/s3929438/all_features_desktop_100_latest_final_all.csv')
#print(data_prep(all_users)[0])