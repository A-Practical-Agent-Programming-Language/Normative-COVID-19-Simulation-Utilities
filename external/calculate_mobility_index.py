#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 23:37:03 2020

@author: Samarth Swarup

This script is essentially the same as calculate_socdist_index.py. It just
accepts input in a different format, e.g.:

date,COUNTYFP,GyrationRadiusKm
2020-01-01,1,46.84
2020-01-02,1,57.62
2020-01-01,3,12.15
2020-01-02,3,15.84
...
"""

import argparse
import pandas as pd
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-i', '--inputfile', help='must specify an input filename', required=True)
requiredNamed.add_argument('-o', '--outputfile', help='must specify an output filename', required=True)

parser.add_argument('-v', '--verbose', help='print extra information while executing', action="store_true")
parser.add_argument('-g', '--debug', help='print debugging information', action="store_true")

args = parser.parse_args()
inputfile = args.inputfile
outputfile = args.outputfile

# For debugging:
# inputfile='gyr_timeline_by_county_ignoring_singletons.csv'
# outputfile='va_county_mobility_index.csv'

df = pd.read_csv(inputfile)
in_df = df.pivot_table(values='GyrationRadiusKm', index='date', columns='COUNTYFP')
in_df.reset_index(inplace=True)
#The resulting format of in_df is:
#COUNTYFP,date,1,3,...820,830,840
#0,2020-01-01,46.85,19.15,...
#1,2020-01-02,57.62,20.61,...
#Note that the column labeled COUNTYFP is actually the index now; it does not contain county FIPS codes any more
#The county FIPS codes are now the columns of the dataframe, after the date column.

num_regions = in_df.shape[1] - 1 #Columns correspond to regions, but the first one is the date; the index column (COUNTYFP) is not counted in in_df.shape[1]

averages = np.zeros((7, num_regions), dtype=float) #Row are days of the week, columns are regions
counts = np.zeros((7, num_regions)) #Row are days of the week, columns are regions
for index, row in in_df.iterrows():
    if row['date'] < '2020-03-01':
        dayofweek = index % 7
        radii = np.delete(row.values,0).astype(float) #All columns except date
        averages[dayofweek] += radii
        counts[dayofweek] += 1
        
averages /= counts
out_df = pd.DataFrame(columns=in_df.columns)
cols = in_df.columns.tolist()
cols.remove('date')

for index, row in in_df.iterrows():
    if row['date'] > '2020-02-29':
        dayofweek = index % 7
        radii = np.delete(row.values,0)
        percent_reduction = (radii - averages[dayofweek])/averages[dayofweek] * 100.0
        percent_reduction_dict = dict(zip(cols, percent_reduction))
        percent_reduction_dict['date'] = row['date']
        out_df = out_df.append(percent_reduction_dict, ignore_index=True)
        
out_df.to_csv(outputfile, index=False)

