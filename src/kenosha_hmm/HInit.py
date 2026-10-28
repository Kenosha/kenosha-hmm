# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 02:12:26 2022

@author: gianl
"""

import pandas as pd
import numpy as np

from kenosha_hmm._frames import prepend_index_level

WORD_ID_COL = 'IndexWord'

def segment_one_word(word, num_states = 3):

    out_word = word.copy()

    ### subdivide the word evenly into states
    state_allocation = []

    remaining_length = len(out_word)
    remaining_states = num_states
    for state in range(1, num_states + 1): ### emitting states start at 1!

        frames_per_state = round(remaining_length / remaining_states)
        stateList = [state] * frames_per_state

        remaining_length -= frames_per_state
        remaining_states -= 1

        state_allocation.extend(stateList)

    ### add the info to the dataframe
    out_word['State'] = state_allocation

    ### and plug it into the index
    out_word = out_word.reset_index().set_index([*out_word.index.names, 'State'])

    return out_word

def find_transition_occurences(word, num_states = 3):

    out_word = word.copy()

    try:
        stateArr = out_word.index.get_level_values('State')
    except:
        stateArr = out_word['State']

    stateArr = [0, *stateArr, num_states + 1] ### add in entry (0) and exit (NUM_STATES + 1) state.

    from_state = stateArr[:-1]
    to_state = stateArr[1:]
    state_changes = np.array([from_state, to_state]).T ### -> shape: WORD_LENGTH (= TRANSITIONS) x 2

    unique, counts = np.unique(state_changes, return_counts = True, axis = 0)

    transMat = np.zeros([num_states + 2, num_states + 2], dtype = int)

    for from_to_pair, number_of_transitions in zip(unique, counts):
        i, j = from_to_pair
        transMat[i, j] = number_of_transitions

    return transMat

def find_transition_matrix(words, num_states = 3):

    sequences = words.index.get_level_values(WORD_ID_COL).unique()

    transMats = np.array([find_transition_occurences(words.loc[idx], num_states = num_states) for idx in sequences])
    all_transMat = transMats.sum(axis = 0)

    rowSum = all_transMat.sum(axis = 1)
    A = np.divide(all_transMat, rowSum[:, np.newaxis],
                  out=np.zeros_like(all_transMat, dtype=float),
                  where=rowSum[:, np.newaxis] != 0)

    ### it can happen that the row sum is zero -> if the state is never occupied non-terminally
    ### always happens with the exit state, btw -> there are no transitions away from it.
    A = np.nan_to_num(A)

    return A

# def get_parameters(words, num_states = 3, variance_floor = 0.01,
#                    cov_how = 'mean'):
#     A = find_transition_matrix(words, num_states = num_states)

#     ### within each word, the mean and covariance within each state
#     avgs = words.groupby(WORD_ID_COL).apply(lambda x: x.groupby('State').mean())
#     covs = words.groupby(WORD_ID_COL).apply(lambda x: x.groupby('State').cov())

#     avg = avgs.groupby('State', sort=True).mean() ### State we want sorted, from left-to-right

#     if cov_how == 'mean':
#         cov = covs.groupby(['State', 'Var'], sort=False).mean() ### to prevent the variable names to be sorted
#     elif cov_how == 'median':
#         cov = covs.groupby(['State', 'Var'], sort=False).median()
#     elif cov_how == 'joint_mean':
#         ### the average of covariances within each word within each state, across states -> joint for all states!
#         joint_cov = covs.groupby('Var').mean()
#         cov = pd.concat([prepend_index_level(joint_cov, 'State', state) for state in range(1, num_states + 1)])
#     elif cov_how == 'global_mean':
#         ### the average of covariances within each word.
#         glob_cov = words.groupby(WORD_ID_COL).cov().groupby('Var').mean()
#         cov = pd.concat([prepend_index_level(glob_cov, 'State', state) for state in range(1, num_states + 1)])

#     ### now, to ensure that state is also sorted here...
#     # cov = cov.sort_index(level = 'State') ## buggy: also sorts Var...

#     ### variance floor
#     floor_values = variance_floor * words.var()
#     for state in avg.index.get_level_values('State').unique():
#         for var in floor_values.index:
#             cov.loc[([state], var), var] = cov.loc[([state], var), var].clip(lower = floor_values[var])

#     ### now, to ensure that state is also sorted here...
#     # cov = cov.sort_index(level = 'State') ## buggy: also sorts Var...
#     cov = pd.concat([cov.loc[[state]] for state in avg.index.get_level_values('State').unique()])

#     return A, avg, cov

def get_parameters(words, num_states, variance_floor = None, cov_global = False):
    A = find_transition_matrix(words, num_states = num_states)

    denominator = words.groupby('State').count()
    avg = words.groupby('State').sum() / denominator

    dist = words - avg
    def cov_by_state(dist_by_state):
        variables = dist_by_state.columns
        cov_vals = np.einsum('iv, iw -> vw', dist_by_state.values, dist_by_state.values)
        return pd.DataFrame(cov_vals, index = variables, columns = variables)

    cov_numerator = dist.groupby('State').apply(cov_by_state)

    if not cov_global:
        cov = cov_numerator / denominator
    else:
        cov = cov_by_state(dist) / words.count()
        cov = pd.concat([prepend_index_level(cov, 'State', state) for state in range(1, num_states + 1)])

    return A, avg, cov


def resolve_viterbi_grid(grid, final_state):
    grid_backwards = reversed(grid) ### first item belongs to the second-to-last state that was occupied
                                    ### last belongs to the first state that was occupied
                                    ### the elements express the likeliest move
                                    ### an element of [1, 1, 3] says that
                                        # if we landed in state 1, we likely came from state 1.
                                        # if we landed in state 2, we likely came from state 1.
                                        # if we landed in state 3, we likely came from state 3.

    stateList = [final_state]
    for likeliest_move in grid_backwards:
        landed_in = stateList[-1]
        came_from = likeliest_move[landed_in]
        stateList.append(came_from)

    return list(reversed(stateList))



def viterbi_segment(densFrm, A):
    sequences = densFrm.index.get_level_values(WORD_ID_COL).unique()

    ### Get the log quantities ###
    log_densFrm = np.log(densFrm)

    with np.errstate(divide='ignore'):
        logA = np.log(A)
    log_transitions_from_initial = logA[0, 1:-1]
    log_transitions_within_states = logA[1:-1, 1:-1]
    log_transition_to_exit = logA[1:-1, -1]

    seq = sequences[0]
    seqList = []
    max_vals = []
    for seq in sequences:

        logFrm = log_densFrm.loc[[seq]]
        logArr = logFrm.values ### Shape: N x S

        ### initialize
        log_density_initial, log_densities = logArr[0], logArr[1:]
        phi = np.array([log_transitions_from_initial + log_density_initial]) ### Shape: 1 x S
        stateArr = []

        ### go through recursive algo
        for log_density in log_densities:
            last_phi_broadcast = phi[-1][:, np.newaxis] ### so the phi can be added to the log transitions.
            viterbi_step = last_phi_broadcast + log_transitions_within_states
            max_state, max_val = viterbi_step.argmax(axis = 0), viterbi_step.max(axis = 0)

            new_phi = max_val + log_density

            phi = np.append(phi, [new_phi], axis = 0)
            stateArr.append(max_state)

        ### last step -> movement into exit state!
        viterbi_step = phi[-1] + log_transition_to_exit
        max_state, max_val = viterbi_step.argmax(), viterbi_step.max()

        ### resolve the grid backward by looking at final state, and then finding second-to-last, etc.
        stateList = resolve_viterbi_grid(stateArr, max_state)

        seqList.extend(stateList)
        max_vals.append(max_val)

    outFrm = log_densFrm.index.to_frame(index = False)
    outFrm['State'] = seqList

    ### turn the statelist back into a one indexed thing -> the zeroth index is the entry state!!!
    outFrm['State'] += 1

    return outFrm, sum(max_vals)

def HInit(words, num_states = 3,
          stop_iteration_threshold = 10 ** -8,
          min_iterations = 15, max_iterations = 100,
          density_floor = 10 ** -100,
          cov_global = False):

    from kenosha_hmm.utils import stop_condition, calculate_emission_densities

    ### Initialize a segmentation by evenly distributing word across states
    words = words.groupby(WORD_ID_COL, group_keys = False).apply(segment_one_word, num_states = num_states)
    state_distro = words.index.get_level_values('State').value_counts().sort_index()

    ### Calculate parameters
    A, avg, cov = get_parameters(words, num_states = num_states, cov_global = cov_global)

    best_A, best_avg, best_cov = A, avg, cov
    best_state_distro = state_distro

    curr_log_likelihood = None
    prev_log_likelihood = None
    best_log_likelihood = None
    num_iterations = 1
    while not stop_condition(curr_log_likelihood, prev_log_likelihood, num_iterations,
                             relative_change_threshold = stop_iteration_threshold,
                             min_iterations = min_iterations, max_iterations = max_iterations):

        prev_log_likelihood = curr_log_likelihood

        try:
            ### With new parameters, new densities
            densFrm = calculate_emission_densities(words, avg, cov,
                                                   density_floor = density_floor)
        except np.linalg.LinAlgError:
            lklStr = round(best_log_likelihood, 2) if best_log_likelihood is not None else None
            print(f'Best Likelihood: {lklStr}')
            print('Implied State Distro:')
            print(best_state_distro)
            return best_A, best_avg, best_cov

        ### Reassign states to observations.
        words.index = words.index.droplevel('State') ### we drop the states -> they will now be recalculated.
        state_sequences, curr_log_likelihood = viterbi_segment(densFrm, A)
        words.index = pd.MultiIndex.from_frame(state_sequences)
        state_distro = words.index.get_level_values('State').value_counts().sort_index()

        ### Recalculate parameters
        A, avg, cov = get_parameters(words, num_states = num_states, cov_global = cov_global)

        ### Save the parameters of the best iteration so far
        if best_log_likelihood is None:
            best_A, best_avg, best_cov = A, avg, cov
            best_log_likelihood = curr_log_likelihood
            best_state_distro = state_distro
        else:
            if curr_log_likelihood > best_log_likelihood:
                best_A, best_avg, best_cov = A, avg, cov
                best_log_likelihood = curr_log_likelihood
                best_state_distro = state_distro

        num_iterations += 1

    else:
        lklStr = round(best_log_likelihood, 2) if best_log_likelihood is not None else None
        print(f'Best Likelihood: {lklStr}')
        print('Implied State Distro:')
        print(best_state_distro)
        return best_A, best_avg, best_cov


def compare_model_lkl_and_gini(lkl1, gini1, lkl2, gini2,
                               rel_gini_weight = 1):
    ### note: an absolute model score is not possible since lkl is a relative measure
    ### and incommensurable with gini

    if lkl1 is None:
        lkl_component = 1
    else:
        lkl_component = (lkl2 - lkl1) / (-lkl1) ### relative improvement of likelihood

    if gini1 is None:
        gini_component = -1
    else:
        gini_component = (gini2 - gini1) / ((1 - gini1) * (1 - gini2)) ## relative worsening of gini coeff.
    ## derivation -> the above expression is equivalent to the difference between (1 / (1 - gini))
    ## which is a severity score that punishes large gini coefficients.

    ### calculate a comp score: positive, if model 2 is better. negative, if 1 is better.
    comp_score = lkl_component - rel_gini_weight * gini_component
    return comp_score

def HInit_robust(words, num_states = 3,
                 stop_iteration_threshold = 10 ** -8,
                 min_iterations = 15, max_iterations = 100,
                 density_floor = 10 ** -100,
                 rel_gini_weight = 1,
                 variance_floor = 0.01,
                 cov_how = 'mean'):
    ### Compared to HInit, here, we try to robustify against small sample initializations
    ### The issue is that with small samples, the state space degenerates -> states become depopulated.
    ### To avoid this, we select results based on log likelihood but also based on the variety of states
    ### This is done by calculating the average gini coefficient of state distribution across words.
    ### Note that convergence by log likelihood is still allowed.
    ### In this case, we likely have no degeneration (to be checked).
    ### However, if the state distribution becomes too narrow or the degenerates to a singleton, we step in.

    def gini(wordFrm):
        state_distribution = wordFrm.index.get_level_values('State').value_counts()
        num_states = len(state_distribution)

        distVals = state_distribution.values
        pairwise_diffs = np.abs(distVals[:, np.newaxis] - distVals[np.newaxis:, ])

        return pairwise_diffs.sum() / (2 * num_states * (num_states - 1) * distVals.mean())

    from kenosha_hmm.utils import stop_condition, calculate_emission_densities

    ### Initialize a segmentation by evenly distributing word across states
    words = words.groupby(WORD_ID_COL, group_keys = False).apply(segment_one_word, num_states = num_states)
    curr_gini = words.groupby(WORD_ID_COL, group_keys = False).apply(gini).mean()

    curr_log_likelihood = None
    prev_log_likelihood = None
    best_log_likelihood = None
    best_gini = None
    num_iterations = 1
    while not stop_condition(curr_log_likelihood, prev_log_likelihood, num_iterations,
                             min_iterations = min_iterations, max_iterations = max_iterations,
                             relative_change_threshold = stop_iteration_threshold):

        ### Calculate parameters
        A, avg, cov = get_parameters(words, num_states = num_states,
                                     variance_floor = variance_floor,
                                     cov_how = cov_how)

        # Save the parameters of the best iteration so far
        new_model_relative_score = compare_model_lkl_and_gini(best_log_likelihood, best_gini,
                                                              curr_log_likelihood, curr_gini,
                                                              rel_gini_weight = rel_gini_weight)

        if new_model_relative_score > 0:
            best_A, best_avg, best_cov = A, avg, cov
            best_log_likelihood = curr_log_likelihood
            best_gini = curr_gini

        try:
            ### With parameters, new densities
            densFrm = calculate_emission_densities(words, avg, cov,
                                                   density_floor = density_floor)
            ### Reassign states to observations.
            words.index = words.index.droplevel('State') ### we drop the states -> they will now be recalculated.
            prev_log_likelihood = curr_log_likelihood
            state_sequences, curr_log_likelihood = viterbi_segment(densFrm, A)
            words.index = pd.MultiIndex.from_frame(state_sequences)
            curr_gini = words.groupby(WORD_ID_COL, group_keys = False).apply(gini).mean()

        except Exception as e:
            lklStr = round(best_log_likelihood, 2) if best_log_likelihood is not None else None
            giniStr = round(best_gini, 2)
            print(f'    ...Stopping: Encountered error: {e}')
            print(f'Best Likelihood: {lklStr}')
            print(f'Best State Gini: {giniStr}')
            return best_A, best_avg, best_cov

        else:
            num_iterations += 1

    else:
        lklStr = round(best_log_likelihood, 2) if best_log_likelihood is not None else None
        giniStr = round(best_gini, 2)
        print(f'Best Likelihood: {lklStr}')
        print(f'Best State Gini: {giniStr}')
        return best_A, best_avg, best_cov
