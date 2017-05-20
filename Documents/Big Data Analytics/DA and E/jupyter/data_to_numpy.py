#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:26:55 2017

@author: tvzyl
"""

import numpy as np
import pandas as pd
from rpy2.robjects import r
from rpy2.robjects import default_converter
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface import NULL

freq_lookup = {1:'D',4:'QS',12:'MS',52:'W',365.15:'AS'}
freq_lookup_period_index = {1:'D',4:'Q',12:'M',52:'W',365.15:'A'}
#"dj"?
file_names = ["melsyd", "a10","fuel", "ausbeer","elec", "milk", "dj", "usconsumption", "austa"]

pandas2ri.activate()

r['library']('fpp')

with localconverter(default_converter + pandas2ri.converter) as cv:    
    for file_name in file_names:
        is_ts = bool(r("is.ts(%s)"%file_name))
        values = r["%s"%file_name]
        if is_ts:
            start = r("start(%s)"%file_name)
            end = r("end(%s)"%file_name)
            frequency = r("frequency(%s)"%file_name)[0]
            freq = freq_lookup[frequency]
            freq_period = freq_lookup_period_index[frequency]
            columns = r("colnames(%s)"%file_name)
            columns = list(columns) if not isinstance(columns, type(NULL)) else [file_name]
            if (start == (1.,1.)).all():
                index = pd.PeriodIndex(start='0001', periods=end[0]*end[1], freq=freq_period)
            else:
                try:
                    dates = r("paste(as.Date(%s))"%file_name)
                    td = pd.to_datetime(dates[1])-pd.to_datetime(dates[0])
                    if td.days>365 and frequency ==1:
                        freq='A-JAN'
                        index = pd.DatetimeIndex(start=dates[0], periods=dates.shape[0], freq=freq)
                    else:
                        index = pd.DatetimeIndex(start=dates[0], end=dates[-1], freq=freq)
                except :
                    if freq == 'W':
                        dates_0 = pd.datetime(int(start[0]),1,1) + pd.DateOffset(weeks=start[1])
                        dates_1 = pd.datetime(int(end[0]),1,1) + pd.DateOffset(weeks=end[1])
                        index = pd.DatetimeIndex(start=dates_0, end=dates_1, freq=freq)
                    else:
                        raise            
            df = pd.DataFrame(values, columns=columns,  index=index)
        elif isinstance(values, pd.DataFrame):
            df = values
        else:
            df = pd.DataFrame(values)
        df.to_pickle('data/%s.pkl'%file_name)
        df.to_csv('data/%s.csv'%file_name)
        exec("%s=df"%file_name)
    
    