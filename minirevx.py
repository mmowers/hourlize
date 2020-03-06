import pandas as pd
import numpy as np
import sklearn.cluster as sc
import pdb
import datetime
import os
import h5py
import json
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
    df_sc = df_sc.groupby(group_cols, sort=False).apply(get_bin, bin_col, num_bins, method)
    df_sc = df_sc.reset_index(drop=True).sort_values('sc_gid')
    print('Done adding bins: '+ str(datetime.datetime.now() - startTime))
    return df_sc

def get_bin(df_in, bin_col, num_bins, method):
    df = df_in.copy()
    ser = df[bin_col]
    #If we have less unique points than num_bins, we simply group the points with the same values.
    if ser.unique().size <= num_bins:
        bin_ser = ser.rank(method='dense')
        df['bin'] = bin_ser.values
    elif method == 'kmeans':
        nparr = ser.to_numpy().reshape(-1,1)
        kmeans = sc.KMeans(n_clusters=num_bins, random_state=0).fit(nparr)
        bin_ser = pd.Series(kmeans.labels_)
        #but kmeans doesn't necessarily label in order of increasing value because it is 2D,
        #so we replace labels with cluster centers, then rank
        kmeans_map = pd.Series(kmeans.cluster_centers_.flatten())
        bin_ser = bin_ser.map(kmeans_map).rank(method='dense')
        df['bin'] = bin_ser.values
    elif method == 'equal_capacity':
        #using a manual method instead of pd.cut because i want the first bin to contain the
        #first sc point regardless, even if its capacity is more than the capacity of the bin,
        #and likewise for other bins, so i don't skip any bins.
        orig_index = df.index
        df.sort_values(by=[bin_col], inplace=True)
        cumcaps = df['capacity'].cumsum().tolist()
        totcap = df['capacity'].sum()
        vals = df[bin_col].tolist()
        bins = []
        curbin = 1
        for i,v in enumerate(vals):
            bins.append(curbin)
            if cumcaps[i] >= totcap*curbin/num_bins:
                curbin += 1
        df['bin'] = bins
        df = df.reindex(index=orig_index) #we need the same index ordering for apply to work.
    return df

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
    df_sc_agg = df_sc.groupby([cf.reg_col,'class','bin']).agg(aggs)
    print('Done aggregating supply curve: '+ str(datetime.datetime.now() - startTime))
    return df_sc_agg

def get_profiles(df_sc, h5_path, h5_dset, id_col, weight_col, ts_path, rep_prof_sel):
    print('Getting average profiles...')
    startTime = datetime.datetime.now()
    df_ts = pd.read_csv(ts_path, low_memory=False)
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    #get unique combinations of region and class
    df_reg_col = df_sc[[cf.reg_col,'class']].drop_duplicates().reset_index(drop=True)
    with h5py.File(h5_path, 'r') as h5:
        #iniitialize avgs_arr and reps_arr with the right dimensions
        avgs_arr = np.zeros((8760,len(df_reg_col)))
        reps_arr = avgs_arr.copy()
        reps_idx = []
        #get idxls, the index of the profiles, which excludes half hour for pv, e.g.
        times = h5['time_index'][:].astype('datetime64')
        time_df = pd.DataFrame({'datetime':times})
        time_df = pd.merge(left=time_df, right=df_ts, on='datetime', how='left', sort=False)
        idxls = time_df[time_df['timeslice'].notnull()].index.tolist()
        for i,r in df_reg_col.iterrows():
            print('region=' + str(r[cf.reg_col]) + ' class=' + str(r['class']))
            df_rc = df_sc[(df_sc[cf.reg_col] == r[cf.reg_col]) & (df_sc['class'] == r['class'])].copy()
            df_rc = df_rc.reset_index(drop=True)
            idls = df_rc[id_col].tolist()
            wtls = df_rc[weight_col].tolist()
            tzls = df_rc['timezone'].tolist()
            tzls = [int(t) for t in tzls]
            if df_rc[id_col].dtype is object:
                idls = [json.loads(l) for l in idls]
                wtls = [json.loads(l) for l in wtls]
                #We will need to flatten idls and wtls, which means we may need to turn tzls into a list of lists too
                #with duplication. For h5 retrieval we also need the ids sorted...
                #idls = ''.join(idls).replace('[','').replace(']',',').replace(' ','').strip(',').split(',')
                #wtls = ''.join(wtls).replace('[','').replace(']',',').replace(' ','').strip(',').split(',')
                #idls = [int(n) for n in idls]
                #wtls = [int(n) for n in wtls]
            if len(idls) != len(wtls):
                print('IDs and weights have different length!')
            arr = h5[h5_dset][:,idls]
            #reduce elements to on the hour using idxls
            arr = arr[idxls,:]
            #Convert to local time and start at 1am instead of 12am, ie roll by an additional 1.
            arr = arr.T
            for n in range(len(arr)):
                arr[n] = np.roll(arr[n], tzls[n] - 1)
            #Take weighted average and add to avgs_arr
            avg_arr = np.average(arr, axis=0, weights=wtls)
            avgs_arr[:,i] = avg_arr
            #Now find the profile in arr that is most representative, ie has the minimum error
            if rep_prof_sel == 'rmse':
                errs = np.sqrt(((arr-avg_arr)**2).mean(axis=1))
            elif rep_prof_sel == 'ave':
                errs = abs(arr.sum(axis=1) - avg_arr.sum())
            min_idx = np.argmin(errs)
            reps_arr[:,i] = arr[min_idx]
            reps_idx.append(idls[min_idx]) #this needs to change for pv
    print('Done getting average profiles: '+ str(datetime.datetime.now() - startTime))
    return df_reg_col, avgs_arr, reps_arr, reps_idx

if __name__== "__main__":
    df_supply_curve = get_df_sc_filtered(cf.supply_curve_path, cf.filter_cols, cf.test_mode, cf.test_filters)
    df_supply_curve = classify(df_supply_curve, cf.resource_class_path)
    df_supply_curve = binnify(df_supply_curve, cf.bin_group_cols, cf.bin_param, cf.bin_num, cf.bin_method)
    output_raw_sc(df_supply_curve, cf.output_dir, cf.output_prefix)
    df_agg_supply_curve = aggregate_sc(df_supply_curve)
    df_agg_supply_curve.to_csv(cf.output_dir + cf.output_prefix + '_supply_curve.csv')
    df_reg_col, avg_prof, rep_prof, rep_idx  = get_profiles(df_supply_curve, cf.profile_h5_path, cf.profile_h5_dset, cf.profile_id_col,
                                              cf.profile_weight_col, cf.timeslice_path, cf.rep_profile_select)
    pdb.set_trace()
