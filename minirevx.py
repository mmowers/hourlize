import pandas as pd
import numpy as np
import sklearn.cluster as sc
import pdb
import datetime
import os
import shutil
import h5py
import json
import config as cf

this_dir_path = os.path.dirname(os.path.realpath(__file__))

def get_df_sc_filtered(sc_path, reg_col, filter_cols={}, test_mode=False, test_filters={}):
    print('Reading supply curve inputs and filtering...')
    startTime = datetime.datetime.now()
    df = pd.read_csv(sc_path, dtype={reg_col:int}, low_memory=False)
    for k in filter_cols.keys():
        df = df[df[k].isin(filter_cols[k])].copy()
    if test_mode:
        for k in test_filters.keys():
            df = df[df[k].isin(test_filters[k])].copy()
    df['region'] = df[reg_col]
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

def binnify(df_sc, bin_group_cols, bin_col, bin_num, bin_method):
    print('Adding bins...')
    startTime = datetime.datetime.now()
    df_sc = df_sc.groupby(bin_group_cols, sort=False).apply(get_bin, bin_col, bin_num, bin_method)
    df_sc = df_sc.reset_index(drop=True).sort_values('sc_gid')
    print('Done adding bins: '+ str(datetime.datetime.now() - startTime))
    return df_sc

def get_bin(df_in, bin_col, bin_num, bin_method):
    df = df_in.copy()
    ser = df[bin_col]
    #If we have less than or equal unique points than bin_num, we simply group the points with the same values.
    if ser.unique().size <= bin_num:
        bin_ser = ser.rank(method='dense')
        df['bin'] = bin_ser.values
    elif bin_method == 'kmeans':
        nparr = ser.to_numpy().reshape(-1,1)
        kmeans = sc.KMeans(n_clusters=bin_num, random_state=0).fit(nparr)
        bin_ser = pd.Series(kmeans.labels_)
        #but kmeans doesn't necessarily label in order of increasing value because it is 2D,
        #so we replace labels with cluster centers, then rank
        kmeans_map = pd.Series(kmeans.cluster_centers_.flatten())
        bin_ser = bin_ser.map(kmeans_map).rank(method='dense')
        df['bin'] = bin_ser.values
    elif bin_method == 'equal_capacity':
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
            if cumcaps[i] >= totcap*curbin/bin_num:
                curbin += 1
        df['bin'] = bins
        df = df.reindex(index=orig_index) #we need the same index ordering for apply to work.
    df['bin'] = df['bin'].astype(int)
    return df

def output_raw_sc(df_sc, out_dir, out_prefix):
    print('Outputting raw csv...')
    startTime = datetime.datetime.now()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df_sc.to_csv(out_dir + out_prefix + '_supply_curve_raw.csv', index=False)
    print('Done outputting raw csv: '+ str(datetime.datetime.now() - startTime))

def aggregate_sc(df_sc):
    print('Aggregating supply curve...')
    startTime = datetime.datetime.now()
    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df_sc.loc[x.index, 'capacity'])
    aggs = {'capacity': 'sum', 'trans_cap_cost':wm, 'dist_mi':wm }
    df_sc_agg = df_sc.groupby(['region','class','bin']).agg(aggs)
    print('Done aggregating supply curve: '+ str(datetime.datetime.now() - startTime))
    return df_sc_agg

def get_profiles(df_sc, profile_path, profile_dset, profile_id_col, profile_weight_col,
                 timeslice_path, to_local, to_1am, rep_profile_method, out_dir, out_prefix):
    print('Getting average profiles...')
    startTime = datetime.datetime.now()
    df_ts = pd.read_csv(timeslice_path, low_memory=False)
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    #get unique combinations of region and class
    df_rep = df_sc[['region','class']].drop_duplicates().sort_values(by=['region','class']).reset_index(drop=True)
    num_profiles = len(df_rep)
    with h5py.File(profile_path, 'r') as h5:
        #iniitialize avgs_arr and reps_arr with the right dimensions
        avgs_arr = np.zeros((8760,num_profiles))
        reps_arr = avgs_arr.copy()
        reps_idx = []
        timezones = []
        #get idxls, the index of the profiles, which excludes half hour for pv, e.g.
        times = h5['time_index'][:].astype('datetime64')
        time_df = pd.DataFrame({'datetime':times})
        time_df = pd.merge(left=time_df, right=df_ts, on='datetime', how='left', sort=False)
        idxls = time_df[time_df['timeslice'].notnull()].index.tolist()
        t0 = datetime.datetime.now()
        for i,r in df_rep.iterrows():
            t1 = datetime.datetime.now()
            df_rc = df_sc[(df_sc['region'] == r['region']) & (df_sc['class'] == r['class'])].copy()
            df_rc = df_rc.reset_index(drop=True)
            idls = df_rc[profile_id_col].tolist()
            wtls = df_rc[profile_weight_col].tolist()
            tzls = df_rc['timezone'].tolist()
            tzls = [int(t) for t in tzls]
            if df_rc[profile_id_col].dtype is object:
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
            t2 = datetime.datetime.now()
            arr = h5[profile_dset][:,idls]
            t3 = datetime.datetime.now()
            #reduce elements to on the hour using idxls
            arr = arr[idxls,:]
            arr = arr.T
            if to_local is True:
                for n in range(len(arr)):
                    arr[n] = np.roll(arr[n], tzls[n])
            #to start at 1am instead of 12am, roll by an additional 1.
            if to_1am is True:
                arr = np.roll(arr, -1, axis=1)
            #Take weighted average and add to avgs_arr
            avg_arr = np.average(arr, axis=0, weights=wtls)
            avgs_arr[:,i] = avg_arr
            #Now find the profile in arr that is most representative, ie has the minimum error
            if rep_profile_method == 'rmse':
                errs = np.sqrt(((arr-avg_arr)**2).mean(axis=1))
            elif rep_profile_method == 'ave':
                errs = abs(arr.sum(axis=1) - avg_arr.sum())
            min_idx = np.argmin(errs)
            reps_arr[:,i] = arr[min_idx]
            reps_idx.append(idls[min_idx]) #Does this need to change for pv?
            timezones.append(tzls[min_idx]) #Does this need to change for pv?
            t4 = datetime.datetime.now()
            frac = (i+1)/num_profiles
            pct = round(frac*100)
            tthis = round((t4 - t1).total_seconds(),2)
            th5 = round((t3 - t2).total_seconds(),2)
            ttot = (t4 - t0).total_seconds()
            mtot, stot = divmod(round(ttot), 60)
            tlft = ttot*(1- frac)/frac
            mlft, slft = divmod(round(tlft), 60)
            print(str(pct)+'%'+
                  '\treg='+str(r['region'])+
                  '\tcls='+str(r['class'])+
                  '\tt='+str(tthis)+'s'+
                  '\th5= '+str(th5)+'s'+
                  '\ttot='+str(mtot)+'m,'+str(stot)+'s'+
                  '\tlft='+str(mlft)+'m,'+str(slft)+'s')

        #scale the data as necessary
        if 'scale_factor' in h5[profile_dset].attrs.keys():
            reps_arr_out = reps_arr.copy()
            scale = h5[profile_dset].attrs['scale_factor']
            reps_arr = reps_arr / scale
            avgs_arr = avgs_arr / scale
        else:
            reps_arr_out = (reps_arr*1000).round().astype('uint16')
    df_rep['rep_gen_gid'] = reps_idx
    df_rep['timezone'] = timezones
    print('Done getting average profiles: '+ str(datetime.datetime.now() - startTime))
    #output profiles to h5 files
    print('Outputting profiles...')
    startTime = datetime.datetime.now()
    out_file = out_dir + out_prefix + '_hourly_cf.h5'
    with h5py.File(out_file, 'w') as f:
        f.create_dataset('rep_profiles_0', data=reps_arr_out)
        f.create_dataset('meta', data=df_rep.to_records(index=False))
    print('Done outputting profiles: '+ str(datetime.datetime.now() - startTime))
    return df_rep, avgs_arr, reps_arr

if __name__== '__main__':
    df_sc = get_df_sc_filtered(cf.sc_path, cf.reg_col, cf.filter_cols, cf.test_mode, cf.test_filters)
    df_sc = classify(df_sc, cf.class_path)
    df_sc = binnify(df_sc, cf.bin_group_cols, cf.bin_col, cf.bin_num, cf.bin_method)
    output_raw_sc(df_sc, cf.out_dir, cf.out_prefix)
    df_sc_agg = aggregate_sc(df_sc)
    df_sc_agg.to_csv(cf.out_dir + cf.out_prefix + '_supply_curve.csv')
    df_rep, avgs_arr, reps_arr = get_profiles(df_sc, cf.profile_path, cf.profile_dset, cf.profile_id_col,
        cf.profile_weight_col, cf.timeslice_path, cf.to_local, cf.to_1am, cf.rep_profile_method, cf.out_dir, cf.out_prefix)
    #Save input files and config to output
    shutil.copy2(this_dir_path + '/minirevx.py', cf.out_dir)
    shutil.copy2(this_dir_path + '/config.py', cf.out_dir)
    shutil.copy2(cf.timeslice_path, cf.out_dir)
    shutil.copy2(cf.class_path, cf.out_dir)