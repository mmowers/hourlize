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
logger.info('Load Logger setup!')

def setup(this_dir_path, out_dir, timeslice_path, calibrate_path, ba_frac_path):
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
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
    shutil.copy2(this_dir_path + 'load.py', out_dir)
    shutil.copy2(this_dir_path + 'config.py', out_dir)
    shutil.copy2(timeslice_path, out_dir)
    shutil.copy2(calibrate_path, out_dir)
    shutil.copy2(ba_frac_path, out_dir)

def calc_outputs(load_source, ba_timezone_path, calibrate_path, ba_frac_path, multiyear, select_year, to_local, truncate_leaps):
    logger.info('Calculating...')
    startTime = datetime.datetime.now()
    df_EI = pd.read_csv(load_source + 'EI_pca_load.csv', low_memory=False, index_col='time', parse_dates=True)
    df_WI = pd.read_csv(load_source + 'WI_pca_load.csv', low_memory=False, index_col='time', parse_dates=True)
    df_ERCOT = pd.read_csv(load_source + 'ERCOT_pca_load.csv', low_memory=False, index_col='time', parse_dates=True)
    df_hr = pd.concat([df_EI, df_WI, df_ERCOT], axis=1)
    #Add logic for testmode?
    if to_local is True:
        #Shift from UTC to local standard time according to timezone
        df_tz = pd.read_csv(ba_timezone_path, low_memory=False)
        shifts = dict(zip(df_tz['ba'], df_tz['timezone']))
        for ba in df_hr:
            df_hr[ba] = np.roll(df_hr[ba], shifts[ba])
    if not multiyear:
        #Remove other years' data
        df_hr = df_hr.loc[str(select_year) + '-01-01':str(select_year) + '-12-31']
    if truncate_leaps is True:
        #Remove December 31 for leap years
        df_hr = df_hr[~((df_hr.index.year % 4 == 0) & (df_hr.index.month == 12) & (df_hr.index.day == 31))]
    logger.info('Done calculating: '+ str(datetime.datetime.now() - startTime))
    return df_hr

def save_outputs(df_hr, out_dir):
    logger.info('Saving outputs...')
    startTime = datetime.datetime.now()
    df_hr.to_csv(out_dir + 'full_hourly_load_input.csv')
    logger.info('Done saving outputs: '+ str(datetime.datetime.now() - startTime))

if __name__== '__main__':
    startTime = datetime.datetime.now()
    this_dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'
    out_dir = this_dir_path + 'out/load_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + '/'
    setup(this_dir_path, out_dir, cf.timeslice_path, cf.calibrate_path, cf.ba_frac_path)
    df_hr = calc_outputs(cf.load_source, cf.ba_timezone_path, cf.calibrate_path, cf.ba_frac_path,
                              cf.multiyear, cf.select_year, cf.to_local, cf.truncate_leaps)
    save_outputs(df_hr, out_dir)
    logger.info('All done! total time: '+ str(datetime.datetime.now() - startTime))
