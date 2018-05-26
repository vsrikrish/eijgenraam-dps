import sys
import os
import xarray as xr
import pandas as pd
import numpy as np

def pareto_efficient(objs):
    eff = np.ones(objs.shape[0], dtype=bool)
    for i, val in enumerate(objs):
        if eff[i]:
            eff[eff] = np.any(objs[eff] <= val, axis=1)
    return eff
    
case = sys.argv[1]
path = 'solutions/{}/'.format(case)
files = os.listdir(path)

var_list = []
obj_list = []
for file in files:
    ds = xr.open_dataset(os.path.join(path,file))
    df = ds.to_dataframe()
    df2 = df[df.index.get_level_values('variable') == 0].reset_index('variable').unstack()['Objectives']
    df3 = df[df.index.get_level_values('objective') == 'Damages'].reset_index('objective').unstack()['Variables']
    var_list.append(df3)
    obj_list.append(df2)

dat = pd.concat(obj_list)
dat.reset_index(drop=True, inplace=True)
objs = dat.values
eff = pareto_efficient(objs)
eff_s = pd.Series(eff)

dat2 = pd.concat(var_list)
dat2.reset_index(drop=True, inplace=True)
par_obj = dat[eff_s]
par_var = dat2[eff_s]
xr.Dataset.from_dataframe(par_obj).to_netcdf('{}-obj.nc'.format(case))

new_name = {i: str(i) for i in range(par_var.shape[1])}
xr.Dataset.from_dataframe(par_var).rename(new_name).to_netcdf(os.path.join(path, '{}-var.nc'.format(case)))
