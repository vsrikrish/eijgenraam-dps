from __future__ import division

import math
import array
import numpy as np
import scipy.stats
import six

# class for Eijgenraam model
class Eijgenraam(object):
    # initialize function
    # parameters that are set to a default of None have values given in Eijgenraam et al (2012) and will
    # default to values associated with the passed dike ring. (Other than h0, which is calculated from the alpha
    # and p0 parameters).
    #
    # investment cost functions are 'exp' or 'quad', where:
    #   exp: (c + bu) exp(a(h+u))
    #   quad: a(h+u)^2 + bu + c
    def __init__(self,
                 inv_fun='exp',  # investment cost function
                 n_years=300,  # number of years
                 ring=16,  # Dike ring for this model
                 fail_stand=None,  # failure standard
                 p0=None,  # initial flooding probability (before t=0)
                 v0=None,  # initial economic value at risk (before t=0)
                 alpha=None,  # exponential distribution parameter for maximum surge (only used to initialize H0)
                 eta=None,  # structural increase in water level (cm/year)
                 zeta=None,  # increase in loss per year
                 a=None,  # investment cost parameter, used for quadratic investment function (Eq. 7)
                 b=None,  # investment cost parameter, used for either quadratic (Eq 7) or exponential (Eq. 10)
                 c=None,  # investment cost parameter, used for either quadratic (Eq 7) or exponential (Eq. 10),
                 h0=None,
                 delta=0.04,  # discount rate, mentioned in Section 2.2
                 gamma=0.035,
                 # growth rate; paper says this is taken from government report, but no indication of actual value
                 rho=0.015):  # risk-free rate, mentioned in Section 2.2

        self.inv_fun = inv_fun
        self.n_years = n_years
        # Parameters pulled from the paper describing each dike ring
        params = ("c", "b", "a", "alpha", "eta", "zeta", "V0", "P0", "max_Pf")
        raw_data = {
            10: (16.6939, 0.6258, 0.0014, 0.033027, 0.320, 0.003774, 1564.9, 0.00044, 1 / 2000),
            11: (42.6200, 1.7068, 0.0000, 0.032000, 0.320, 0.003469, 1700.1, 0.00117, 1 / 2000),
            15: (125.6422, 1.1268, 0.0098, 0.050200, 0.760, 0.003764, 11810.4, 0.00137, 1 / 2000),
            16: (324.6287, 2.1304, 0.0100, 0.057400, 0.760, 0.002032, 22656.5, 0.00110, 1 / 2000),
            22: (154.4388, 0.9325, 0.0066, 0.070000, 0.620, 0.002893, 9641.1, 0.00055, 1 / 2000),
            23: (26.4653, 0.5250, 0.0034, 0.053400, 0.800, 0.002031, 61.6, 0.00137, 1 / 2000),
            24: (71.6923, 1.0750, 0.0059, 0.043900, 1.060, 0.003733, 2706.4, 0.00188, 1 / 2000),
            35: (49.7384, 0.6888, 0.0088, 0.036000, 1.060, 0.004105, 4534.7, 0.00196, 1 / 2000),
            38: (24.3404, 0.7000, 0.0040, 0.025321, 0.412, 0.004153, 3062.6, 0.00171, 1 / 1250),
            41: (58.8110, 0.9250, 0.0033, 0.025321, 0.422, 0.002749, 10013.1, 0.00171, 1 / 1250),
            42: (21.8254, 0.4625, 0.0019, 0.026194, 0.442, 0.001241, 1090.8, 0.00171, 1 / 1250),
            43: (340.5081, 4.2975, 0.0043, 0.025321, 0.448, 0.002043, 19767.6, 0.00171, 1 / 1250),
            44: (24.0977, 0.7300, 0.0054, 0.031651, 0.316, 0.003485, 37596.3, 0.00033, 1 / 1250),
            45: (3.4375, 0.1375, 0.0069, 0.033027, 0.320, 0.002397, 10421.2, 0.00016, 1 / 1250),
            47: (8.7813, 0.3513, 0.0026, 0.029000, 0.358, 0.003257, 1369.0, 0.00171, 1 / 1250),
            48: (35.6250, 1.4250, 0.0063, 0.023019, 0.496, 0.003076, 7046.4, 0.00171, 1 / 1250),
            49: (20.0000, 0.8000, 0.0046, 0.034529, 0.304, 0.003744, 823.3, 0.00171, 1 / 1250),
            50: (8.1250, 0.3250, 0.0000, 0.033027, 0.320, 0.004033, 2118.5, 0.00171, 1 / 1250),
            51: (15.0000, 0.6000, 0.0071, 0.036173, 0.294, 0.004315, 570.4, 0.00171, 1 / 1250),
            52: (49.2200, 1.6075, 0.0047, 0.036173, 0.304, 0.001716, 4025.6, 0.00171, 1 / 1250),
            53: (69.4565, 1.1625, 0.0028, 0.031651, 0.336, 0.002700, 9819.5, 0.00171, 1 / 1250)}
        data = {i: {k: v for k, v in zip(params, raw_data[i])} for i in six.iterkeys(raw_data)}
        if fail_stand is None:
            self.fail_stand = data[ring]["max_Pf"]
        else:
            self.fail_stand = fail_stand
        if p0 is None:
            self.p0 = data[ring]["P0"]  # flood probability just before t=0
        else:
            self.p0 = p0
        if v0 is None:
            self.v0 = data[ring]["V0"]  # economic value at risk before t=0
        else:
            self.v0 = v0
        if alpha is None:
            self.alpha = data[ring]["alpha"]  # exponential surge distribution parameter (not used with GEV)
        else:
            self.alpha = alpha
        if eta is None:
            self.eta = data[ring]["eta"]
        else:
            self.eta = eta
        if zeta is None:
            self.zeta = data[ring]["zeta"]
        if b is None:
            self.b = data[ring]["b"]
        else:
            self.b = b
        if c is None:
            self.c = data[ring]["c"]
        else:
            self.c = c
        if a is None:
            self.a = data[ring]["a"]
        else:
            self.a = a
        self.rho = rho
        self.gamma = gamma
        self.delta = delta
        self.delta1 = self.delta + self.rho
        if h0 is None:
            # the initial height is calculated as the expected value based on an exponential distribution
            self.h0 = math.ceil(10 * (1 / self.alpha) * math.log(1 / self.p0) / 100) / 10
        else:
            self.h0 = h0
        self.h = array.array('d', (0,) * self.n_years)  # array for dike heights over time
        self.u = array.array('d', (0,) * self.n_years)  # array for heightenings for each year
        self.fb = array.array('d', (0,) * self.n_years) # array for freeboard heights
        self.bf = array.array('d', (0,) * self.n_years) # array for buffer heights

    # Returns the required standard of protection
    def get_protection_standard(self):
        return self.fail_stand

    # Assigns the buffer and freeboard decision variables for all simulation years and decides on heightening
    def compute_heightenings(self,
                             dps_case,  # 'local' only or 'remote'
                             basis_fun,  # 'gaussian' or 'cubic'
                             w,  # numpy array of weights for DPS basis functions
                             x,  # numpy array of center parameter for DPS basis functions
                             r,  # numpy array of radius parameters for DPS basis functions
                             sl_window,  # length of window for sea level statistics
                             sea_levels,  # numpy array of annual block maxima tidal records (mean sea levels + surges),
                             ice_window=None,  # if dps_case is 'remote', pass length of ice sheet window
                             ice_contrib=None):  # if dps_case is 'remote', pass ice sheet contributions as a numpy array

        sl_length = np.size(sea_levels)
        # we assume the last year of the sea level/ice sheet contribution series is the same as the last year
        # for the problem, but the record may go back earlier than the starting year, even including the
        # start of the first window
        length_diff = sl_length - self.n_years

        if dps_case == 'local':
            state_vars = np.zeros(shape=(self.n_years, 2))
        else:
            state_vars = np.zeros(shape=(self.n_years, 4))

        # for each year, calculate the decision variables and decide on heightening
        for t in range(self.n_years):
        # find slope and sum-of-squares residuals for sea levels over window
            sl_fit = np.polyfit(np.arange(sl_window), sea_levels[length_diff + t - sl_window:length_diff + t], 1,full=True)
            if dps_case == 'remote':
                ice_fit = np.polyfit(np.arange(ice_window), ice_contrib[length_diff + t - ice_window:length_diff + t], 1, full=True)
                state_vars[t, :] = np.array([sl_fit[0].item(0), np.sqrt(sl_fit[1].item()), ice_fit[0].item(0), np.sqrt(ice_fit[1].item())])
            elif dps_case == 'local':
                state_vars[t, :] = np.array([sl_fit[0][0], sl_fit[1][0]])
            else:
                raise Exception('Incorrect state variable option! Must be "local" or "remote".')
            if np.size(state_vars[t, :]) != x.shape[1]:
                raise Exception('Wrong number of decision variables!')

            # calculate decision variables from state variables
            # freeboard heights and buffers should not be negative
            if basis_fun == 'cubic':
                rad_nz = [np.nonzero(r[d]) for d in
                          range(r.shape[0])]  # find which radii are zero so we don't divide by zero
                self.fb[t] = np.maximum(
                    np.dot(np.power(np.abs(state_vars[t, rad_nz[0]] - x[0, rad_nz[0]]) / r[0, rad_nz[0]], 3)[0], w[0, rad_nz[0]][0]), 0)
                self.bf[t] = np.maximum(
                    np.dot(np.power(np.abs(state_vars[t, rad_nz[1]] - x[1, rad_nz[1]]) / r[1, rad_nz[1]], 3)[0], w[1, rad_nz[1]][0]), 0)
            elif basis_fun == 'Gaussian':
                rad_nz = [np.nonzero(r[d]) for d in
                          range(r.shape[0])]  # find which radii are zero so we don't divide by zero
                self.fb[t] = np.maximum(
                    np.dot(np.exp(-np.power((state_vars[t, rad_nz[0]] - x[0, rad_nz[0]]) / r[0, rad_nz[0]], 2))[0], w[0, rad_nz[0]][0]), 0)
                self.bf[t] = np.maximum(
                    np.dot(np.exp(-np.power((state_vars[t, rad_nz[1]] - x[1, rad_nz[1]]) / r[1, rad_nz[1]], 2))[0], w[1, rad_nz[1]][0]), 0)
            elif basis_fun == 'quadratic':
                self.fb[t] = np.maximum(
                    w[0] + np.sum(r[0]*state_vars[t, :] + x[0]*np.power(state_vars[t, :],2)),0)
                self.bf[t] = np.maximum(
                    w[1] + np.sum(r[1] * state_vars[t, :] + x[1] * np.power(state_vars[t, :],2)), 0)
            else:
                raise Exception('Basis function form not supported! Must be "cubic," "Gaussian," or "quadratic."')

            # calculate whether a heightening occurs based on the decision variables and the observed sea level
            if t == 0:
                self.h[t] = self.h0
            else:
                self.h[t] = self.h[t - 1] + self.u[t - 1]

            test_height = self.h[t] - self.bf[t]
            if sea_levels[length_diff + t] > test_height:
                safe_height = sea_levels[length_diff + t] - test_height
                # compute heightening; this can only occur in increments of 10 cm
                self.u[t] = math.ceil(10*(safe_height + self.fb[t]))/10
            else:
                self.u[t] = 0

        return state_vars

    # Compute the investment cost to increase the dike height
    def investment_cost(self,
                        cost_fun='exp'):  # can be 'exp' or 'quad'

        if cost_fun == 'exp':
            inv_cost = (self.c + self.b * 100*np.array(self.u)) * np.exp(self.a * 100*(np.array(self.h) - self.h0)) * (np.array(self.u) > 0)
        elif cost_fun == 'quad':
            inv_cost = self.a * np.pow((100*(np.array(self.h) - self.h0)), 2) + self.b * 100 * np.array(self.u) + self.c

        # discount investment cost and return
        return inv_cost*np.array([np.power(1+self.delta, -t) for t in range(self.n_years)])


    # Compute the expected damages from the heightening series. Also returns the maximum exceedance probability.
    def expected_damages(self,
                         sea_levels):

        sl_length = np.size(sea_levels)
        length_diff = sl_length - self.n_years
        # what is the height surges have to exceed to overtop the dike? Subsidence is assumed to be linear
        # with respect to the rate eta.
        effective_height = np.array([self.h[t] - self.eta*t/100 for t in range(self.n_years)])
        # if the height exceeds the dike level, the entire ring is wiped out per Eijgenraam's assumptions
        floods = np.greater(sea_levels[length_diff:], effective_height)
        # compute the economic value at risk behind the dike ring (equation (2)).
        v = np.array([self.v0 * np.exp(self.gamma * t + 100*self.zeta * (self.h[t] - self.h0)) for t in range(self.n_years)])
        # compute the losses from flooding
        s = np.array([v[t] * int(floods[t]) for t in range(self.n_years)])
        # discount s for NPV
        s_disc = s*np.array([np.power(1+self.delta1,-t) for t in range(self.n_years)])

        return [s_disc, floods]

    def exceed_prob(self,
                    mean_sea_levels,
                    surge_params,
                    surge_stationary=True,
                    temps = None):

        sl_length = np.size(mean_sea_levels)

        # we assume the last year of the sea level/ice sheet contribution series is the same as the last year
        # for the problem, but the record may go back earlier than the starting year, even including the
        # start of the first window
        length_diff = sl_length - self.n_years

        # compute the exceedance probability
        # what is the height surges have to exceed to overtop the dike? Subsidence is assumed to be linear
        # with respect to the rate eta.
        current_buffer = np.array([self.h[t] - self.eta * t / 100 - mean_sea_levels.iloc[length_diff + t] for t in range(self.n_years)])
        # compute the exceedance probability of the current buffer
        if surge_stationary:
            exceed_prob = scipy.stats.genextreme.sf(x=current_buffer * 1000, c=-surge_params['shape'],
                                                    loc=surge_params['loc'],
                                                    scale=surge_params['scale'])
        else:
            exceed_prob = scipy.stats.genextreme.sf(x=current_buffer * 1000, c=-surge_params['shape'],
                                                    loc=surge_params['loc0'] + surge_params['loc1'] * temps.iloc[length_diff:],
                                                    scale=surge_params['scale'])
        return exceed_prob[1:]

# generate synthetic tidal records from mean sea level simulations and surge distributions.
# surge distributions can be stationary or not; temperature simulations must be included if
# they are stationary. We only consider a non-stationary location parameter for now.
def synth_tidal_records(mean_sea_levels,  # numpy array of mean sea levels for a particular SOW
                        GEV_params, # tuple of GEV parameters, (loc, scale, shape). if non-stationary, loc is the coefficient of temp.
                        surge_stationary=True,  # is the surge distribution stationary?
                        temp=None):  # pass a numpy array of global mean temperatures if the surge distribution is non-stationary.

    if surge_stationary:
        # scipy.stats.genextreme has a different sign convention than R for GEV shape parameters
        ann_max_surge = scipy.stats.genextreme.rvs(c=-GEV_params['shape'], loc=GEV_params['loc'], scale=GEV_params['scale'],
                                                   size=len(mean_sea_levels))
        return mean_sea_levels + ann_max_surge/1000
    else:
        ann_max_surge = scipy.stats.genextreme.rvs(c=-GEV_params['shape'], loc=(GEV_params['loc1'] * temp)+GEV_params['loc0'],
                                                   scale=GEV_params['scale'],size=len(mean_sea_levels))
        return mean_sea_levels + ann_max_surge/1000

# function to use for pandas apply on each SOW for generating synthetic tidal records.
def apply_synth_tidal(SOW,
                     surge_stationary=True):

    if surge_stationary:
        return synth_tidal_records(SOW.LocalSeaLevel, SOW.SurgeParams, surge_stationary)
    else:
        return synth_tidal_records(SOW.LocalSeaLevel, SOW.SurgeParams, surge_stationary, SOW.Temperature)
