from __future__ import print_function

import eijgenraam as ei
from configparser import ConfigParser, ExtendedInterpolation
import pandas as pd
import numpy as np
import xarray as xr
import sys

# class to read configuration files.
class PropertiesParser(ConfigParser):

    def check_values(self):

        # process DPS properties
        try:
            self.has_section('DPS')
        except:
            raise Exception('Need "DPS" section for DPS parameters!')
        # identify the number of state variables based on the DPS case
        try:
            dps_info = self.get('DPS', 'information')
        except:
            raise Exception('Must specify DPS information as "remote" or "local"!')
        if dps_info == 'local':
            self.set('DPS', 'num_state_vars', '2')
        elif dps_info == 'remote':
            self.set('DPS', 'num_state_vars', '4')
        else:
            raise Exception('Invalid DPS Case! Must be "remote" or "local".')

        try:
            basis_func = self.get('DPS', 'basis')
        except:
            raise Exception('Must specify DPS basis function as "cubic" or "Gaussian"!')
        if basis_func not in ['cubic', 'Gaussian', 'quadratic']:
            raise Exception('Invalid basis function form! Must be "cubic" or "Gaussian".')

        try:
            self.get('DPS', 'sl window')
        except:
            raise Exception('Need tide gauge observation window!')
        ice_window = self.get('DPS', 'ice window', fallback=None)
        if dps_info == 'remote' and ice_window is None:
            raise Exception('Need ice contribution window for remote model!')

        # check simulation parameters
        if self.get('Simulation', 'investment') not in ['exp', 'quad']:
            raise Exception('Investment function must be "exp" or "quad"!')

        # is the number of SOWs positive?
        num_sows = self.getint('Simulation', 'num sows')
        if num_sows <= 0:
            raise Exception('Number of SOWs must be positive!')

        # resolve file paths
        self.set('Paths', 'sea_level_path', self.get('Paths', 'sea level path'))
        self.set('Paths', 'surge_path', self.get('Paths', 'surge path'))
        self.set('Paths', 'output_path', self.get('Paths', 'output path'))

        self.set('Paths', 'runtime_path', self.get('Paths', 'runtime path'))

    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
        return d


def read_config_file(filename):
    cfg_parser = PropertiesParser()
    cfg_parser.read(filename)
    cfg_parser.check_values()
    cfg_args = cfg_parser.as_dict()
    # paths get resolved, so we don't need to keep the original literal values
    cfg_args['Paths'].pop('sea level path')
    cfg_args['Paths'].pop('surge path')
    # convert config parameter names and values to lower case if they are strings
    cfg_args = {sec_name: {k.replace(' ', '_'): v for k, v in sec_opt.items()}
                for sec_name, sec_opt in cfg_args.items()}
    return cfg_args


# function to set up tide series from file inputs
def sample_tide_data(sea_level_files, # this should be a list, even if it's a single file
                     surge_file,
                     surge_stationary,
                     first_year,
                     num_sows,
                     dps_information):

    # read in mean sea level projection files and convert to a list of pandas dataframes
    sea_level_projections = [xr.open_dataset(file, engine='netcdf4').to_dataframe() for file in sea_level_files]
    sea_level_proj_unstack = [proj.unstack(level=1) for proj in sea_level_projections]
    # concatenate into a single ensemble
    sea_level_ens_unstack = pd.concat(sea_level_proj_unstack, ignore_index=True)
    sea_level_ensemble = sea_level_ens_unstack.stack()

    # normalize mean sea levels and temperatures (if appropriate) to the reference year (the first year of the simulation)
    tidal_variables = ['LocalSeaLevel']
    if not surge_stationary:
        tidal_variables += ['Temperature']
    if dps_information == 'remote':
        ice_variables = ['AIS', 'GIS']
    else:
        ice_variables = None

    # check if the appropriate variables were passed in the sea level projections
    if not set(tidal_variables).issubset(sea_level_ensemble.columns):
        raise Exception('Non-stationary surges require temperature projections!')
    if dps_information == 'remote':
        if not set(ice_variables).issubset(sea_level_ensemble.columns):
            raise Exception('Remote assimilation requires ice contribution projections!')

    sea_level_ensemble[tidal_variables] = sea_level_ensemble[tidal_variables].subtract(sea_level_ensemble[tidal_variables].xs(first_year, level='time_proj'), level=0)
    sea_level_samples = sea_level_ensemble.unstack().sample(num_sows, replace=True).reset_index(drop=True).stack()

    # import surge distributions
    if surge_stationary:
        surge_fit = pd.read_csv(surge_file, names=['loc', 'scale', 'shape'], usecols=[1, 2, 3], skiprows=1)
        # sample surge SOWs
        surge_samples = surge_fit.sample(num_sows, replace=True).reset_index(drop=True)
        # for each SOW, generate tide gauge series
        tide_samples = pd.concat([sea_level_samples['LocalSeaLevel'].unstack('time_proj'), surge_samples], axis=1,
                                 keys=['LocalSeaLevel', 'SurgeParams'])
    else:
        surge_fit = pd.read_csv(surge_file, names=['loc0', 'loc1', 'scale', 'shape'],
                                usecols=[1, 2, 3, 4], skiprows=1)
        surge_samples = surge_fit.sample(num_sows, replace=True).reset_index(drop=True)
        tide_samples = pd.concat([sea_level_samples['LocalSeaLevel'].unstack('time_proj'),
                                  sea_level_samples['Temperature'].unstack('time_proj'),
                                  surge_samples], axis=1, keys=['LocalSeaLevel', 'Temperature', 'SurgeParams'])

    # compute synthetic tide series and return as numpy array
    tide_series = tide_samples.apply(ei.apply_synth_tidal, surge_stationary=surge_stationary, axis=1).as_matrix()

    # if remote information is used, return ice sheet observations as numpy array
    if dps_information == 'remote':
        ice_series = (sea_level_samples['AIS'] + sea_level_samples['GIS']).unstack().as_matrix()
    else:
        ice_series = None

    return {'tide_params': tide_samples, 'tide_series': tide_series, 'ice_contrib': ice_series}


# this function is called by Borg for the optimization. It takes as an input the decision variables
# from Borg, sets up the states of the world, loops over the states of the world, and then returns
# the objective values. It also takes in keyword args to specify whether the problem is remote or local
# and what type of basis functions to use.
def eijgenraam_sim(borg_vars,
                   config):

    # construct model object
    eij = ei.Eijgenraam(n_years=2101 - int(config['Simulation']['first_year']),
                        ring=int(config['Simulation']['ring']),
                        h0=5.5)
    # reshape the decision variables from Borg
    # dec_vars[0] will be the weights
    # dec_vars[1] will be the centers
    # dec_vars[2] will be the scaling factors
    if config['DPS']['basis'] == 'quadratic':
        temp_vars = np.reshape(np.asarray(borg_vars[2:]),
                                        [2, 2, int(config['DPS']['num_state_vars'])])
        dec_vars = np.array([np.append(np.asarray(borg_vars[:2]),
                                       np.zeros([1, 2*int(config['DPS']['num_state_vars'])-2])),
                             temp_vars[0],
                             temp_vars[1]
                             ])
    else:
        dec_vars = np.reshape(np.asarray(borg_vars), [3, 2, int(config['DPS']['num_state_vars'])])

    # get constraint (protection standard)
    pstand = eij.get_protection_standard()

    # read number of SOWs
    num_sows = int(config['Simulation']['num_sows'])
    # read tide series and ice contributions
    tide_params = config['SOWs']['tide_params']
    tide_series = config['SOWs']['tide_series']
    ice_contribution = config['SOWs']['ice_contrib']

    # initialize objective storage arrays
    exp_dam = np.zeros(num_sows)
    inv_cost = np.zeros(num_sows)
    max_exceed = np.zeros(num_sows)
    flood_num = np.zeros(num_sows)

    for i in range(num_sows):
        if config['DPS']['information'] == 'remote':
            eij.compute_heightenings(dps_case=config['DPS']['information'],
                                     basis_fun=config['DPS']['basis'],
                                     w=dec_vars[0],
                                     x=dec_vars[1],
                                     r=dec_vars[2],
                                     sl_window=int(config['DPS']['sl_window']),
                                     sea_levels=tide_series[i],
                                     ice_window=int(config['DPS']['ice_window']),
                                     ice_contrib=ice_contribution[i])
        else:
            eij.compute_heightenings(dps_case=config['DPS']['information'],
                                     basis_fun=config['DPS']['basis'],
                                     w=dec_vars[0],
                                     x=dec_vars[1],
                                     r=dec_vars[2],
                                     sl_window=int(config['DPS']['sl_window']),
                                     sea_levels=tide_series.loc[i].as_matrix())

        inv_cost[i] = np.sum(eij.investment_cost(cost_fun=config['Simulation']['investment']))

        flood_dam = eij.expected_damages(sea_levels=tide_series.loc[i].as_matrix())
        exp_dam[i] = np.sum(flood_dam[0])
        flood_num[i] = np.sum(flood_dam[1])

        # if config['Simulation']['stationary'] == 'true':
        #     exceed_prob = eij.exceed_prob(mean_sea_levels=tide_params['LocalSeaLevel'].iloc[i],
        #                                          surge_params=tide_params['SurgeParams'].iloc[i],
        #                                          surge_stationary=True)
        # else:
        #     exceed_prob = eij.exceed_prob(mean_sea_levels=tide_params['LocalSeaLevel'].iloc[i],
        #                                          surge_params=tide_params['SurgeParams'].iloc[i],
        #                                          surge_stationary=False,
        #                                          temps=tide_params['Temperature'].iloc[i])
        #
        # max_exceed[i] = np.amax(exceed_prob)

    flood_rel = np.sum(flood_num)/(num_sows*eij.n_years)
    # if expected damages or investment costs are infinite or nan, set the objectives and constraints to the worst
    # possible values
    if np.isnan(np.sum(inv_cost)) or np.isinf(np.sum(inv_cost)) or np.isnan(np.sum(exp_dam)) or np.isinf(np.sum(exp_dam)):
        objs = [np.finfo('d').max, np.finfo('d').max, np.finfo('d').max]
        const = [np.finfo('d').min]
    # otherwise return the true objectives
    else:
        objs = [np.mean(inv_cost), np.mean(exp_dam), flood_rel]
        const = [np.minimum(0.001-flood_rel, 0).tolist()]

    return (objs, const)


# function called to set up Borg and the simulation and start the optimization
def optimize_eijgenraam(config_file, run_name):

    import os
    from borg import borg

    # set initial seed
    np.random.seed(100)

    # read configuration file
    config_args = read_config_file(config_file)

    # turn mean sea level file input into a list in case there are multiple files that need to be read in
    sea_level_file_paths = config_args['Paths']['sea_level_path'].split(',')

    # generate tide gauge and (if appropriate) ice contribution SOWs
    config_args['SOWs'] = sample_tide_data(sea_level_files=sea_level_file_paths,
                                   surge_file=config_args['Paths']['surge_path'],
                                   surge_stationary=config_args['Simulation']['stationary'] == 'true',
                                   first_year=int(config_args['Simulation']['first_year']),
                                   num_sows=int(config_args['Simulation']['num_sows']),
                                   dps_information=config_args['DPS']['information'])

    # set up Borg
    # set number of decision variables: three variables per basis function, two policy variables, and the number
    # of state variables depends on the information assimilated into the DPS
    num_state_vars = int(config_args['DPS']['num_state_vars'])
    if config_args['DPS']['basis'] == 'quadratic':
        num_dec_vars = 2*2*num_state_vars+2
    else:
        num_dec_vars = 3*2*num_state_vars
    # set number of objectives: we have two (minimizing investment cost and expected future damages)
    num_objectives = 3
    # set whether the objectives should be maximized or minimized
    objective_directions = [borg.Direction.MINIMIZE,borg.Direction.MINIMIZE,borg.Direction.MINIMIZE]
    # set number of constraints: we just have the one (is the max exceedence probability for the dike ring
    # respected?
    num_constraints = 1

    # set bounds for the decision variables. For now, this is hard-coded.
    # bounds for the weights are between -5 and 5
    # bounds for the centers are between 0 and 300
    # bounds for the scaling factors are between -50 and 50
    if config_args['DPS']['basis'] == 'quadratic':
        dec_var_bounds = [[0, 1] for var in range(2)] + \
                         ([[-0.5, 0.5], [-0.5, .5]] + [[-0.5, 0.5] for var in range(num_state_vars-2)])*2 + \
                         ([[-1, 1], [-1, 1]] + [[-1, 1] for var in range(num_state_vars-2)])*2
    else:
        dec_var_bounds = [[0, 2] for var in range(2*num_state_vars)] + \
                         [[0, 5] for var in range(2*num_state_vars)] + \
                         [[0, 10] for var in range(2*num_state_vars)]

    # set epsilon values for each objective.
    epsilons = [0.1, 0.1, 0.00005]

    # if Borg should run in parallel, start MPI
    if config_args['Borg']['parallel'].lower() == 'true':
        result = borg.Configuration.startMPI()

    for seed in range(int(config_args['Borg']['num_seeds'])):
        borg.Configuration.seed(13*(seed+1))
        # set up Borg problem
        bg = borg.Borg(numberOfVariables=num_dec_vars,
                       numberOfObjectives=num_objectives,
                       numberOfConstraints=num_constraints,
                       function=eijgenraam_sim,
                       config=config_args,
                       bounds=dec_var_bounds,
                       epsilons=epsilons,
                       directions=objective_directions)

        # if parallel is true, run multi-master Borg with global Latin hypercube initialization
        if config_args['Borg']['parallel'].lower() == 'true':
            # set runtime file name
            if 'runtime_path' in config_args['Paths']:
                try:
                    os.makedirs(os.path.join(config_args['Paths']['runtime_path'], run_name))
                except OSError:
                    if not os.path.isdir(os.path.join(config_args['Paths']['runtime_path'], run_name)):
                        raise

                fs_enc = sys.getfilesystemencoding()
                runtime_file = os.path.join(config_args['Paths']['runtime_path'], run_name, 'runtime-%d-{}.txt'.format(seed))
                runtime_file = runtime_file.encode(fs_enc)
                runtime_frequency = int(config_args['Borg']['runtime_freq'])
            else:
                runtime_file = None
                runtime_frequency = None

            result = bg.solveMPI(islands=int(config_args['Borg']['num_islands']),
                                 maxEvaluations=int(config_args['Borg']['num_func_evals']),
                                 maxTime=int(config_args['Borg']['max_time']),
                                 initialization=2,
                                 runtime=runtime_file,
                                 frequency=runtime_frequency)
        else:
            if 'runtime_path' in config_args['Paths']:
                try:
                    os.makedirs(os.path.join(config_args['Paths']['runtime_path'], run_name))
                except OSError:
                    if not os.path.isdir(os.path.join(config_args['Paths']['runtime_path'], run_name)):
                        raise

                fs_enc = sys.getfilesystemencoding()
                runtime_file = os.path.join(config_args['Paths']['runtime_path'], run_name, 'runtime-{}.txt'.format(seed))
                runtime_file = runtime_file.encode(fs_enc)
                runtime_frequency = int(config_args['Borg']['runtime_freq'])
            else:
                runtime_file = None
                runtime_frequency = None

            result = bg.solve(settings={'maxEvaluations': int(config_args['Borg']['num_func_evals']),
                                          'runtimeformrat': 'borg',
                                          'frequency': runtime_frequency,
                                          'runtime_filename': runtime_file})

        # when this seed has finished running, display results and write objective values and decision variables
        # to netcdf file
        if result:
            result.display(separator="\n")
            objs, dec_vars, constr_flag = [], [], []
            for solution in result:
                objs.append(solution.getObjectives())
                dec_vars.append(solution.getVariables())
                constr_flag.append(solution.violatesConstraints())
            soln_idx = [i for i in range(result.size())]
            var_idx = [i for i in range(num_dec_vars)]
            obj_array = xr.DataArray(data=objs, coords=[('index', soln_idx), ('objective', ['Investment','Damages', 'ExceedProb'])],
                                     name='objective values')
            dec_var_array = xr.DataArray(data=dec_vars,
                                         coords=[('index', soln_idx), ('variable', var_idx)],
                                         name='Pareto-optimal decision variables')
            constr_flag_array = xr.DataArray(data=constr_flag,
                                             coords=[('index', soln_idx)],
                                             name='Pareto-optimal constraint violations')
            solution_set = xr.Dataset({"Objectives": obj_array, "Variables": dec_var_array,
                                       "Violations": constr_flag_array})
            try:
                os.makedirs(os.path.join(config_args['Paths']['output_path'], run_name))
            except OSError:
                if not os.path.isdir(os.path.join(config_args['Paths']['output_path'], run_name)):
                    raise

            sol_file_path = os.path.join(config_args['Paths']['output_path'], run_name, 'solution-{}.nc'.format(seed))
            if os.path.isfile(sol_file_path):
                os.remove(sol_file_path)
                solution_set.to_netcdf(sol_file_path, mode='w')
            else:
                solution_set.to_netcdf(sol_file_path, mode='w')

            print('Seed {} complete'.format(seed))

        # stop MPI if parallel
        if config_args['Borg']['parallel'].lower() == 'true':
            borg.Configuration.stopMPI()

if __name__ == '__main__':

    # call function to start optimization
    optimize_eijgenraam(sys.argv[1], sys.argv[2])
