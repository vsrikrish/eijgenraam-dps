from __future__ import print_function

import eijgenraam as ei
import pandas as pd
import numpy as np
import xarray as xr
import sys
from eijgenraam_optim import sample_tide_data

def eijgenraam_sim_mincost(case_name):

    import os

    # set initial seed
    np.random.seed(10)

    valid_eij = ei.Eijgenraam(n_years=2101 - 2017, ring=16, h0=5.5)
#    overall_eij = ei.Eijgenraam(n_years=2101 - 2017, ring=16, h0=5.5)
    pstand = valid_eij.get_protection_standard()
    # get economically efficient solution
    path = 'solutions/{}/'.format(case)
    dps_objs = xr.open_dataset(os.path.join(path, '{}-obj.nc'.format(case_name))).to_dataframe()
    dps_sols = xr.open_dataset(os.path.join(path, '{}-var.nc'.format(case_name))).to_dataframe()
    try:
        valid_min_cost_idx = dps_objs[dps_objs.ExceedProb <= pstand][['Damages', 'Investment']].sum(
            axis=1).argmin()
    except ValueError:
        valid_min_cost_idx = dps_objs.ExceedProb.argmin()
    valid_min_cost_sol = dps_sols.loc[valid_min_cost_idx]
    print(dps_objs.loc[valid_min_cost_idx])
    print(valid_min_cost_sol)

#    overall_min_cost_idx = dps_objs.Objectives.unstack()[['Damages','Investment']].sum(axis=1).argmin()
#    overall_min_cost_sol = dps_sols.loc[overall_min_cost_idx]

    # generate SOWs
    RCPs = [2.6, 4.5, 6.0, 8.5]
    sea_level_file_paths = ['data/BRICK_LSL_NED_RCP{}.nc'.format(str(rcp).replace(".","")) for rcp in RCPs]
    if 'stationary' in case_name:
        surge_file_path = 'data/Delfzijl_gev_stationary.csv'
        surge_stationary = True
    else:
        surge_file_path = 'data/Delfzijl_gev_nonstationary.csv'
        surge_stationary = False

    if 'remote' in case_name:
        dps_info = 'remote'
        num_state_vars = 4
    else:
        dps_info = 'local'
        num_state_vars = 2

    # reshape decision variables
    # dec_vars[0] will be the weights
    # dec_vars[1] will be the centers
    # dec_vars[2] will be the scaling factors

    temp_vars = np.reshape(np.asarray(valid_min_cost_sol[2:]),
                           [2, 2, num_state_vars])
    valid_dec_vars = np.array([np.append(np.asarray(valid_min_cost_sol[:2]),
                                   np.zeros([1, 2 * num_state_vars - 2])),
                         temp_vars[0],
                         temp_vars[1]])

#    temp_vars = np.reshape(np.asarray(overall_min_cost_sol[2:]),
#                           [2, 2, num_state_vars])

#    overall_dec_vars = np.array([np.append(np.asarray(overall_min_cost_sol[:2]),
#                                   np.zeros([1, 2 * num_state_vars - 2])),
#                         temp_vars[0],
#                         temp_vars[1]])

    num_sows = 100000

    sows = [sample_tide_data(sea_level_files=[file],
                             surge_file=surge_file_path,
                             surge_stationary=surge_stationary,
                             first_year=2017,
                             num_sows=num_sows,
                             dps_information=dps_info)
            for file in sea_level_file_paths]

    valid_inv = np.empty(shape=(num_sows, len(RCPs), 2101-2017))
    valid_dam = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    valid_heights = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    valid_freeboard = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    valid_buffer = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    valid_floods = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    valid_state_vars = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017, num_state_vars))

    # overall_inv = np.empty(shape=(num_sows, len(RCPs), 2101-2017))
    # overall_dam = np.empty(shape=(num_sows, len(RCPs), 2101-2017))
    #
    # overall_heights = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    # overall_freeboard = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    # overall_buffer = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    #
    # overall_floods = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))
    #
    # overall_state_vars = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017, num_state_vars))

    tide_heights = np.empty(shape=(num_sows, len(RCPs), 2101 - 2017))

    # simulate sows
    for j in range(len(RCPs)):
        for i in range(num_sows):
            tide_heights[i, j, :] = sows[j]['tide_series'].loc[i][[year for year in range(2017, 2101)]].as_matrix()
            if dps_info == 'remote':
                valid_state_vars[i, j, :, :] = valid_eij.compute_heightenings(dps_case=dps_info,
                                                                              basis_fun='quadratic',
                                                                              w=valid_dec_vars[0],
                                                                              x=valid_dec_vars[1],
                                                                              r=valid_dec_vars[2],
                                                                              sl_window=30,
                                                                              sea_levels=sows[j]['tide_series'].loc[i].as_matrix(),
                                                                              ice_window=10,
                                                                              ice_contrib=sows[j]['ice_contrib'].loc[i].as_matrix())

                # overall_state_vars[i, j, :, :] = overall_eij.compute_heightenings(dps_case=dps_info,
                #                                                                   basis_fun='quadratic',
                #                                                                   w=overall_dec_vars[0],
                #                                                                   x=overall_dec_vars[1],
                #                                                                   r=overall_dec_vars[2],
                #                                                                   sl_window=30,
                #                                                                   sea_levels=sows[j]['tide_series'].loc[i].as_matrix(),
                #                                                                   ice_window=10,
                #                                                                   ice_contrib=sows[j]['ice_contrib'].loc[i].as_matrix())
            else:
                valid_state_vars[i, j, :, :] = valid_eij.compute_heightenings(dps_case=dps_info,
                                                                              basis_fun='quadratic',
                                                                              w=valid_dec_vars[0],
                                                                              x=valid_dec_vars[1],
                                                                              r=valid_dec_vars[2],
                                                                              sl_window=30,
                                                                              sea_levels=sows[j]['tide_series'].loc[i].as_matrix())

                # overall_state_vars[i, j, :, :] = overall_eij.compute_heightenings(dps_case=dps_info,
                #                                                                   basis_fun='quadratic',
                #                                                                   w=overall_dec_vars[0],
                #                                                                   x=overall_dec_vars[1],
                #                                                                   r=overall_dec_vars[2],
                #                                                                   sl_window=30,
                #                                                                   sea_levels=sows[j]['tide_series'].loc[i].as_matrix())

            valid_heights[i, j, :] = valid_eij.h
            valid_freeboard[i, j, :] = valid_eij.fb
            valid_buffer[i, j, :] = valid_eij.bf
            valid_inv[i, j, :] = valid_eij.investment_cost(cost_fun='exp')
            flood_dam = valid_eij.expected_damages(sea_levels=sows[j]['tide_series'].loc[i].as_matrix())
            valid_dam[i, j, :] = flood_dam[0]
            valid_floods[i, j, :] = flood_dam[1]

            # overall_heights[i, j, :] = overall_eij.h
            # overall_freeboard[i, j, :] = overall_eij.fb
            # overall_buffer[i, j, :] = overall_eij.bf
            # overall_inv[i, j, :] = overall_eij.investment_cost(cost_fun='exp')
            # flood_dam = overall_eij.expected_damages(sea_levels=sows[j]['tide_series'].loc[i].as_matrix())
            # overall_dam[i, j, :] = flood_dam[0]
            # overall_floods[i, j, :] = flood_dam[1]

    year_idx = [year for year in range(2017, 2101)]
    sow_idx = [sow for sow in range(num_sows)]
    rcp_idx = [rcp for rcp in RCPs]
    var_idx = [i for i in range(num_state_vars)]

    tide_array = xr.DataArray(data=tide_heights, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
                              name='water height')

    # ovr_ht_array = xr.DataArray(data=overall_heights, coords = [('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)], name='dike height')
    # ovr_fb_array = xr.DataArray(data=overall_freeboard, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
    #                             name='freeboard')
    # ovr_bf_array = xr.DataArray(data=overall_buffer, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
    #                             name='buffer')
    # ovr_inv_array = xr.DataArray(data=overall_inv, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
    #                              name='investment')
    # ovr_dam_array = xr.DataArray(data=overall_dam, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
    #                              name='damages')
    # ovr_fd_array = xr.DataArray(data=overall_floods, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
    #                             name='flood')
    #
    # ovr_sv_array = xr.DataArray(data=overall_state_vars, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx), ('variable', var_idx)],
    #                             name='state variables')
    # ovr_set = xr.Dataset({'Dike Height': ovr_ht_array, 'Freeboard Height': ovr_fb_array, 'Buffer Height': ovr_bf_array,
    #                       'Investment': ovr_inv_array, 'Damages': ovr_dam_array, 'Floods': ovr_fd_array,
    #                       'State': ovr_sv_array, 'Water Height': tide_array})

    val_ht_array = xr.DataArray(data=valid_heights, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)], name='dike height')

    val_fb_array = xr.DataArray(data=valid_freeboard, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
                                name='freeboard')
    val_bf_array = xr.DataArray(data=valid_buffer, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
                                name='buffer')
    val_inv_array = xr.DataArray(data=valid_inv, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
                                name='investment')
    val_dam_array = xr.DataArray(data=valid_dam, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
                                name='damages')
    val_fd_array = xr.DataArray(data=valid_floods, coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx)],
                                name='flood')
    val_sv_array = xr.DataArray(data=valid_state_vars,
                                coords=[('SOW', sow_idx), ('RCP', rcp_idx), ('year', year_idx),
                                        ('variable', var_idx)],
                                name='state variables')
    val_set = xr.Dataset(
        {'Dike Height': val_ht_array, 'Freeboard Height': val_fb_array, 'Buffer Height': val_bf_array,
         'Investment': val_inv_array, 'Damages': val_dam_array, 'Floods': val_fd_array,
         'State': val_sv_array, 'Water Height': tide_array})

    try:
        os.makedirs(os.path.join('sim', case_name))
    except OSError:
        if not os.path.isdir(os.path.join('sim', case_name)):
            raise

    val_file_path = os.path.join('sim', case_name, 'valid.nc')

    if os.path.isfile(val_file_path):
        os.remove(val_file_path)
        val_set.to_netcdf(val_file_path, mode='w')
        print('File written!')
    else:
        val_set.to_netcdf(val_file_path, mode='w')
        print('File written!')

    # ovr_file_path = os.path.join('sim', case_name, 'overall.nc')
    #
    # if os.path.isfile(ovr_file_path):
    #     os.remove(ovr_file_path)
    # else:
    #     ovr_set.to_netcdf(ovr_file_path, mode='w')

if __name__ == '__main__':

    eijgenraam_sim_mincost(sys.argv[1])
