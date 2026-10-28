"""Gaussian emissions and training convergence from the original 2022 implementation.

Author: Gianluca Vagliano. ATMAN-specific dataset and filesystem adapters remain
in ATMAN-v1; this module has no dependency on that project.
"""
import numpy as np
import pandas as pd

def calculate_emission_densities(words, avg, cov, density_floor = None):
    """
    for each frame in each word occurence, we need the likelihood
    of said frame being emitted when in any of the HMM states.
    """

    ### PREPARE SOME HELPER VARS
    states = avg.index.unique()
    num_states = len(states)
    features = list(avg.columns)
    num_vars = len(features)

    ### PREPARE ARRAYS ###
    wordArr = words[features].values ### shape: num_obs, num_vars
    avgArr = avg.values.T  ### shape: num_vars, num_states
    covArr = cov.values.reshape(num_states, num_vars, num_vars) ### shape: num_states, num_vars, num_vars
    dists = wordArr[:, :, np.newaxis] - avgArr[np.newaxis, :, :] ### shape: num_obs, num_vars, num_states

    invArr = []
    detArr = []
    for mat in covArr:
        if any(np.linalg.eigvals(mat) <= 0): ### check for positive semidefiniteness
            print('Covariance matrix singular!')
            raise np.linalg.LinAlgError
        else:
            invArr.append(np.linalg.inv(mat))
            detArr.append(np.linalg.det(mat))

    invArr = np.array(invArr) ### shape: num_states, num_vars, num_vars
    detArr = np.array(detArr) ### shape: num_states

    ### CALCULATE NORMAL DENSITY LIKELIHOOD
    # a) Numerator
    trans_vec_mat_mulStr = 'nis, sij -> njs'
    dist_x_cov_inv = np.einsum(trans_vec_mat_mulStr, dists, invArr)

    vec_vec_mulStr = 'njs, njs -> ns'
    exp_term = np.einsum(vec_vec_mulStr, dist_x_cov_inv, dists)
    numerator = np.exp(-0.5 * exp_term) ### shape: num_obs, num_states

    # b) Denominator
    denominator = (((2 * np.pi) ** num_vars) * detArr) ** 0.5 ### shape: num_states

    # c) Result
    density = numerator / denominator[np.newaxis, :] ### shape: num_obs, num_states

    densityFrm = pd.DataFrame(density, index = words.index, columns = states)

    if density_floor is not None:
        densityFrm = densityFrm.clip(lower = density_floor)

    return densityFrm

def stop_condition(curr_log_likelihood, prev_log_likelihood,
                   num_iterations,
                   relative_change_threshold = 0.0001,
                   min_iterations = 15, max_iterations = 100):
    try:
        round_likelihood = round(curr_log_likelihood, 2)
    except:
        round_likelihood = None

    print(f'Iteration {num_iterations}. Current likelihood: {round_likelihood}...')
    if num_iterations < min_iterations:
        # print(f'    ...Minimum of {min_iterations} not reached!')
        return False
    elif num_iterations >= max_iterations:
        print(f'    ...Stopping: {max_iterations} iterations exceeded.')
        return True
    else:
        if num_iterations is None or prev_log_likelihood is None:
            return False ### don't stop!
        else:
            relative_diff_likelihood = abs((curr_log_likelihood - prev_log_likelihood) / curr_log_likelihood)
            if relative_diff_likelihood < relative_change_threshold:
                print('    ...Stopping: Convergence achieved.')
                return True
            else:
                return False
