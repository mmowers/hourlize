import pandas as pd
import numpy as np
import sklearn.cluster as sc
import pdb
import datetime
import os
import h5py
import config as cf

def get_df_sc_filtered(sc_path, filter_cols={}, test_mode=False, test_filters={}):
    print('Reading supply curve inputs and filtering...')
    startTime = datetime.datetime.now()
    df = pd.read_csv(sc_path, low_memory=False)
    for k in filter_cols.keys():
        df = df[df[k].isin(filter_cols[k])].copy()
    if test_mode:
        for k in test_filters.keys():
            df = df[df[k].isin(test_filters[k])].copy()
    print('Done reading supply curve inputs and filtering: '+ str(datetime.datetime.now() - startTime))
    return df

def classify(df_sc, class_path):
    print('Adding classes...')
    startTime = datetime.datetime.now()
    df_class = pd.read_csv(class_path, index_col='class')
    df_sc['class'] = 'NA'
    for cname, row in df_class.iterrows():
        mask = True
        for col, val in row.items():
            if '|' in val:
                rng = val.split('|')
                rng = [float(n) for n in rng]
                mask = mask & (df_sc[col] >= min(rng))
                mask = mask & (df_sc[col] < max(rng))
            else:
                mask = mask & (df_sc[col] == val)
        df_sc.loc[mask, 'class'] = cname
    print('Done adding classes: '+ str(datetime.datetime.now() - startTime))
    return df_sc

def binnify(df_sc, group_cols, bin_col, num_bins, method):
    print('Adding bins...')
    startTime = datetime.datetime.now()
    df_sc['bin'] = df_sc.groupby(group_cols, sort=False)[bin_col].transform(get_bin, bin_col, num_bins, method)
    print('Done adding bins: '+ str(datetime.datetime.now() - startTime))
    return df_sc

def get_bin(ser, bin_col, num_bins, method):
    if method == 'kmeans':
        #If we have less unique points than num_bins, we simply group the points with the same values.
        if ser.unique().size < num_bins:
            bin_ser = ser.rank(method='dense')
        else:
            nparr = ser.to_numpy().reshape(-1,1)
            kmeans = sc.KMeans(n_clusters=num_bins, random_state=0).fit(nparr)
            bin_ser = pd.Series(kmeans.labels_)
            #but kmeans doesn't necessarily label in order of increasing value because it is 2D,
            #so we replace labels with cluster centers, then rank
            kmeans_map = pd.Series(kmeans.cluster_centers_.flatten())
            bin_ser = bin_ser.map(kmeans_map).rank(method='dense')
    return bin_ser

def output_raw_sc(df_sc, out_dir, prefix):
    print('Outputting raw csv...')
    startTime = datetime.datetime.now()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_sc.to_csv(out_dir + prefix + '_supply_curve_raw.csv', index=False)
    print('Done outputting raw csv: '+ str(datetime.datetime.now() - startTime))

def aggregate_sc(df_sc):
    print('Aggregating supply curve...')
    startTime = datetime.datetime.now()
    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df_sc.loc[x.index, "capacity"])
    aggs = {'capacity': 'sum', 'trans_cap_cost':wm, 'dist_mi':wm }
    df_sc_agg = df_sc.groupby(['model_region','class','bin']).agg(aggs)
    print('Done aggregating supply curve: '+ str(datetime.datetime.now() - startTime))
    return df_sc_agg

def get_average_profiles(df_sc, h5_path, h5_dset, id_col, weight_col, ts_path):
    df_ts = pd.read_csv(ts_path, low_memory=False)
    with h5py.File(h5_path, 'r') as profile_file:
        pass

if __name__== "__main__":
    df_supply_curve = get_df_sc_filtered(cf.supply_curve_path, cf.filter_cols, cf.test_mode, cf.test_filters)
    df_supply_curve = classify(df_supply_curve, cf.resource_class_path)
    df_supply_curve = binnify(df_supply_curve, cf.bin_group_cols, cf.bin_param, cf.bin_num, cf.bin_method)
    output_raw_sc(df_supply_curve, cf.output_dir, cf.output_prefix)
    df_agg_supply_curve = aggregate_sc(df_supply_curve)
    df_agg_supply_curve.to_csv(cf.output_dir + cf.output_prefix + '_supply_curve.csv')
    #average_profile_arr = get_average_profiles(df_supply_curve, cf.profile_h5_path, cf.profile_h5_dset, cf.profile_id_col, cf.profile_weight_col, cf.timeslice_path)
    pdb.set_trace()
