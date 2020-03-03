import pandas as pd
import numpy as np
import sklearn.cluster as sc
import pdb
import datetime
import os
import config as cf

def filter_df(df, filter_cols):
    for c in filter_cols.keys():
        df = df[df[c].isin(filter_cols[c])].copy()
    return df

def classify(df_sc,df_class):
    print('Adding classes...')
    startTime = datetime.datetime.now()
    df_class = df_class.set_index('class')
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


def aggregate_sc(df_sc):
    print('Aggregating supply curve...')
    startTime = datetime.datetime.now()
    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df_sc.loc[x.index, "capacity"])
    aggs = {'capacity': 'sum', 'trans_cap_cost':wm, 'dist_mi':wm }
    df_sc_agg = df_sc.groupby(['model_region','class','bin']).agg(aggs)
    print('Done aggregating supply curve: '+ str(datetime.datetime.now() - startTime))
    return df_sc_agg

if __name__== "__main__":
    print('Reading supply curve inputs...')
    startTime = datetime.datetime.now()
    df_resource_classes = pd.read_csv(cf.resource_class_path)
    df_supply_curve = pd.read_csv(cf.supply_curve_path, low_memory=False)
    if cf.filter_cols is not None:
        df_supply_curve = filter_df(df_supply_curve,cf.filter_cols)
    if cf.testmode:
        df_supply_curve = df_supply_curve[df_supply_curve['model_region'].isin(cf.testmode_reg)].copy()
    print('Done reading supply curve inputs: '+ str(datetime.datetime.now() - startTime))
    df_supply_curve = classify(df_supply_curve, df_resource_classes)
    df_supply_curve = binnify(df_supply_curve, cf.bin_group_cols, cf.bin_param, cf.bin_num, cf.bin_method)
    print('Outputting raw csv...')
    startTime = datetime.datetime.now()
    if not os.path.exists(cf.output_dir):
        os.makedirs(cf.output_dir)
    df_supply_curve.to_csv(cf.output_dir + cf.output_prefix + '_supply_curve_raw.csv', index=False)
    print('Done outputting raw csv: '+ str(datetime.datetime.now() - startTime))
    df_agg_supply_curve = aggregate_sc(df_supply_curve)
    df_agg_supply_curve.to_csv(cf.output_dir + cf.output_prefix + '_supply_curve.csv')
    pdb.set_trace()
