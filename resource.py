import pandas as pd
import numpy as np
import sklearn.cluster as sc
import pdb
import datetime
import os
import sys
import shutil
import tables
import json
import config as cf
import logging

#Setup logger
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.info('Logger setup!')

def setup(this_dir_path, out_dir, timeslice_path, class_path):
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    #Create output directory, creating backup if one already exists.
    if os.path.exists(out_dir):
        os.rename(out_dir, os.path.dirname(out_dir) + '-archive-'+time)
    os.makedirs(out_dir)

    #Add output file for logger
    fh = logging.FileHandler(out_dir + 'log.txt', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    #Copy inputs to outputs
    shutil.copy2(this_dir_path + 'resource.py', out_dir)
    shutil.copy2(this_dir_path + 'config.py', out_dir)
    shutil.copy2(timeslice_path, out_dir)
    shutil.copy2(class_path, out_dir)

def get_df_sc_filtered(sc_path, reg_col, filter_cols={}, test_mode=False, test_filters={}):
    logger.info('Reading supply curve inputs and filtering...')
    startTime = datetime.datetime.now()
    df = pd.read_csv(sc_path, dtype={reg_col:int}, low_memory=False)
    for k in filter_cols.keys():
        df = df[df[k].isin(filter_cols[k])].copy()
    if test_mode:
        for k in test_filters.keys():
            df = df[df[k].isin(test_filters[k])].copy()
    df['region'] = df[reg_col]
    logger.info('Done reading supply curve inputs and filtering: '+ str(datetime.datetime.now() - startTime))
    return df

def classify(df_sc, class_path):
    logger.info('Adding classes...')
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
    logger.info('Done adding classes: '+ str(datetime.datetime.now() - startTime))
    return df_sc

def binnify(df_sc, bin_group_cols, bin_col, bin_num, bin_method):
    logger.info('Adding bins...')
    startTime = datetime.datetime.now()
    df_sc = df_sc.groupby(bin_group_cols, sort=False).apply(get_bin, bin_col, bin_num, bin_method)
    df_sc = df_sc.reset_index(drop=True).sort_values('sc_gid')
    logger.info('Done adding bins: '+ str(datetime.datetime.now() - startTime))
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
    elif bin_method == 'equal_cap_man':
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
    elif bin_method == 'equal_cap_cut':
        orig_index = df.index
        df.sort_values(by=[bin_col], inplace=True)
        df['cum_cap'] = df['capacity'].cumsum()
        bin_ser = pd.cut(df['cum_cap'], bin_num, labels=False)
        bin_ser = bin_ser.rank(method='dense')
        df['bin'] = bin_ser.values
        df = df.reindex(index=orig_index) #we need the same index ordering for apply to work.
    df['bin'] = df['bin'].astype(int)
    return df

def aggregate_sc(df_sc):
    logger.info('Aggregating supply curve...')
    startTime = datetime.datetime.now()
    # Define a lambda function to compute the weighted mean:
    wm = lambda x: np.average(x, weights=df_sc.loc[x.index, 'capacity'])
    aggs = {'capacity': 'sum', 'trans_cap_cost':wm, 'dist_mi':wm }
    df_sc_agg = df_sc.groupby(['region','class','bin']).agg(aggs)
    logger.info('Done aggregating supply curve: '+ str(datetime.datetime.now() - startTime))
    return df_sc_agg

def get_profiles(df_sc, profile_path, profile_dset, profile_id_col, profile_weight_col,
                 timeslice_path, to_local, rep_profile_method, driver, gather_method):
    logger.info('Getting profiles...')
    startTime = datetime.datetime.now()
    df_ts = pd.read_csv(timeslice_path, low_memory=False)
    df_ts['datetime'] = pd.to_datetime(df_ts['datetime'])
    #get unique combinations of region and class
    df_rep = df_sc[['region','class']].drop_duplicates().sort_values(by=['region','class']).reset_index(drop=True)
    num_profiles = len(df_rep)
    with tables.open_file(profile_path, 'r', driver=driver) as h5:
        #iniitialize avgs_arr and reps_arr with the right dimensions
        avgs_arr = np.zeros((8760,num_profiles))
        reps_arr = avgs_arr.copy()
        reps_idx = []
        timezones = []
        #get idxls, the index of the profiles, which excludes half hour for pv, e.g.
        times = h5.root['time_index'][:].astype('datetime64')
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
            if df_rc[profile_id_col].dtype == object:
                #This means we are using a column whose values represent lists. This happens for pv.
                idls = [json.loads(l) for l in idls]
                wtls = [json.loads(l) for l in wtls]
                #duplicate tzls entries to match up with idls
                for n in range(len(tzls)):
                    tzls[n] = [tzls[n]] * len(idls[n])
                #flatten lists and gather into dataframe
                idls = [item for sublist in idls for item in sublist]
                wtls = [item for sublist in wtls for item in sublist]
                tzls = [item for sublist in tzls for item in sublist]
                df_ids = pd.DataFrame({'idls':idls, 'wtls':wtls, 'tzls':tzls})
                #sum weighting over duplicate ids. For timezone, this is potentially problematic. I'm simply picking the first one...
                #this also ends up sorting by id, important for h5 retrieval:
                df_ids =  df_ids.groupby(['idls'], as_index =False).agg({'wtls':sum, 'tzls':lambda x: x.iloc[0]})
                idls = df_ids['idls'].tolist()
                wtls = df_ids['wtls'].tolist()
                tzls = df_ids['tzls'].tolist()
            if len(idls) != len(wtls):
                logger.info('IDs and weights have different length!')
            t2 = datetime.datetime.now()
            ave_spacing = (max(idls) - min(idls))/len(idls)
            if gather_method == 'slice' or (gather_method == 'smart' and ave_spacing < 250):
                min_idls = min(idls)
                orig_idls_idx = [j - min_idls for j in idls]
                arr = h5.root[profile_dset][:,min_idls:max(idls)+1]
                arr = arr[:, orig_idls_idx].copy()
            else:
                arr = h5.root[profile_dset][:,idls]
            t3 = datetime.datetime.now()
            #reduce elements to on the hour using idxls
            arr = arr[idxls,:]
            arr = arr.T
            if to_local is True:
                for n in range(len(arr)):
                    arr[n] = np.roll(arr[n], tzls[n])
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
            reps_idx.append(idls[min_idx])
            timezones.append(tzls[min_idx])
            t4 = datetime.datetime.now()
            frac = (i+1)/num_profiles
            pct = round(frac*100)
            tthis = round((t4 - t1).total_seconds(),2)
            th5 = round((t3 - t2).total_seconds(),2)
            ttot = (t4 - t0).total_seconds()
            mtot, stot = divmod(round(ttot), 60)
            tlft = ttot*(1- frac)/frac
            mlft, slft = divmod(round(tlft), 60)
            logger.info(str(pct)+'%'+
                  '\treg='+str(r['region'])+
                  '\tcls='+str(r['class'])+
                  '\tt='+str(tthis)+'s'+
                  '\th5= '+str(th5)+'s'+
                  '\ttot='+str(mtot)+'m,'+str(stot)+'s'+
                  '\tlft='+str(mlft)+'m,'+str(slft)+'s')

        #scale the data as necessary
        if 'scale_factor' in h5.root[profile_dset].attrs:
            scale = h5.root[profile_dset].attrs['scale_factor']
            reps_arr = reps_arr / scale
            avgs_arr = avgs_arr / scale
    df_rep['rep_gen_gid'] = reps_idx
    df_rep['timezone'] = timezones
    logger.info('Done getting profiles: '+ str(datetime.datetime.now() - startTime))
    return df_rep, avgs_arr, reps_arr, df_ts

def calc_performance(avgs_arr, reps_arr, df_rep, df_ts, cfmean_type):
    logger.info('Calculate peformance characteristics...')
    startTime = datetime.datetime.now()
    df_cfmean = df_rep[['region','class']].copy()
    df_cfsigma = df_cfmean.copy()
    if cfmean_type == 'ave':
        cfmean_arr = avgs_arr.copy()
    elif cfmean_type == 'rep':
        cfmean_arr = reps_arr.copy()
    ts_ls = df_ts['timeslice'].unique().tolist()
    for ts in ts_ls:
        ts_idx = df_ts[df_ts['timeslice'] == ts].index
        cfmeans = np.mean(cfmean_arr[ts_idx], axis=0)
        df_cfmean[ts] = cfmeans
        #we use reps_arr regardless for standard deviations. Perhaps we should use the average of the standard deviations tho...
        cfsigmas = np.std(reps_arr[ts_idx], ddof=1, axis=0)
        df_cfsigma[ts] = cfsigmas
    df_cfmean['type'] = 'cfmean'
    df_cfsigma['type'] = 'cfsigma'
    df_perf = pd.concat([df_cfmean,df_cfsigma], sort=False).reset_index(drop=True)
    df_perf = pd.melt(df_perf, id_vars=['region','class','type'], value_vars=ts_ls, var_name='timeslice', value_name= 'value')
    df_perf = df_perf.pivot_table(index=['region','class','timeslice'], columns='type', values='value')
    logger.info('Done with performance calcs: '+ str(datetime.datetime.now() - startTime))
    return df_perf

def save_outputs(df_sc, df_sc_agg, df_perf, reps_arr, df_ts, df_rep, out_dir, out_prefix):
    logger.info('Saving outputs...')
    startTime = datetime.datetime.now()
    df_sc.to_csv(out_dir + out_prefix + '_supply_curve_raw.csv', index=False)
    df_sc_agg.to_csv(out_dir + out_prefix + '_supply_curve.csv')
    df_perf.to_csv(out_dir + out_prefix + '_performance.csv')
    #output profiles to h5 file
    out_file = out_dir + out_prefix + '_hourly_cf.h5'
    reps_arr_out = (reps_arr*1000).round().astype('uint16')
    with tables.open_file(out_file, 'w') as h5:
        h5.create_array(h5.root, 'rep_profiles_0', reps_arr_out, 'representative profiles')
        h5.create_array(h5.root, 'time_index', df_ts['datetime'].to_numpy().astype('S'), 'time index of profiles')
        #The following throws a warning because 'class' is one of the column headers.
        h5.create_table(h5.root, 'meta', df_rep.to_records(index=False), 'meta on each profile')
    logger.info('Done saving outputs: '+ str(datetime.datetime.now() - startTime))

if __name__== '__main__':
    startTime = datetime.datetime.now()
    this_dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    out_dir = this_dir_path + 'out/' + cf.out_dir
    setup(this_dir_path, out_dir, cf.timeslice_path, cf.class_path)
    df_sc = get_df_sc_filtered(cf.sc_path, cf.reg_col, cf.filter_cols, cf.test_mode, cf.test_filters)
    df_sc = classify(df_sc, cf.class_path)
    df_sc = binnify(df_sc, cf.bin_group_cols, cf.bin_col, cf.bin_num, cf.bin_method)
    df_sc_agg = aggregate_sc(df_sc)
    df_rep, avgs_arr, reps_arr, df_ts = get_profiles(df_sc, cf.profile_path, cf.profile_dset, cf.profile_id_col,
        cf.profile_weight_col, cf.timeslice_path, cf.to_local, cf.rep_profile_method, cf.driver, cf.gather_method)
    df_perf = calc_performance(avgs_arr, reps_arr, df_rep, df_ts, cf.cfmean_type)
    save_outputs(df_sc, df_sc_agg, df_perf, reps_arr, df_ts, df_rep, out_dir, cf.out_prefix)
    logger.info('All done! total time: '+ str(datetime.datetime.now() - startTime))
