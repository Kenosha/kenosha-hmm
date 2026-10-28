# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:34:32 2022

@author: gianl
"""

import numpy as np
import pandas as pd

######## LOG ARITHMETIC #########
###### (after Mann, 2006) #######
#################################

LOGZERO = -np.inf

# however, for this to work without warning, we have to ignore the dividebyzero warning raised by np.log(0)

# also, we ignore invalid values (except within eln, see below). Usually, these are add errors.

def eexp(x):
    return np.exp(x) ###

def eln(x):
    with np.errstate(divide='ignore', invalid='raise'): ### this way, we catch negative values in log
        return np.log(x)

def elnsum(eln_x, eln_y):
    # Stable for extreme likelihoods and -inf; no process-wide NumPy changes.
    return np.logaddexp(eln_x, eln_y)

def elnproduct(eln_x, eln_y):
    with np.errstate(invalid='ignore'):
        return np.where((eln_x == LOGZERO) | (eln_y == LOGZERO), LOGZERO, eln_x + eln_y)

##############################
#### LOG BAUM WELCH ALGO #####
##############################


WORD_ID_COL = 'IndexWord'


def log_forward(seqArr, A):
    ### ALGORITHM 5 in Tobias P. Mann

    transitions_from_initial = A[0, 1:-1]
    transitions_within_states = A[1:-1, 1:-1]
    transitions_to_exit = A[1:-1, -1]

    emitting_states = transitions_within_states.shape[0]

    density_initial, densities = seqArr[0], seqArr[1:]

    ### Initial Conditions ###
    # a) entry state (1.17 in HTK book)
    # elna_initial_entry = eln(1) # for sure in the entry state at beginning
    # b) emitting states (1.18 in HTK Book)
    elna_initial_emit = elnproduct(eln(transitions_from_initial), eln(density_initial))
    # c) exit state (not in HTK book)
    # elna_initial_exit = eln(0) # can not be in exit state at the beginning
    elnaArr = np.array([elna_initial_emit])

    ### Recursion ###
    # note that the recursion is done vectorized for all three states.
    # hence, we have one nested fewer loops (2 instead of 3) in the recursion compared to Mann (2006)
    for densArr in densities:
        last_elna = elnaArr[-1]
        logalpha = np.array([LOGZERO] * emitting_states) ### initialize across emitting states

        for emitting_state in range(emitting_states):
            recur_alpha = last_elna[emitting_state]
            recur_trans = eln(transitions_within_states[emitting_state])

            logalpha = elnsum(logalpha,
                              elnproduct(recur_alpha,
                                         recur_trans))

        elna = elnproduct(logalpha, eln(densArr))
        elnaArr = np.append(elnaArr, [elna], axis = 0)

    ### Final Condition ###
    last_elna = elnaArr[-1]
    logalpha = LOGZERO

    for emitting_state in range(emitting_states):
        recur_alpha = last_elna[emitting_state]
        recur_trans = eln(transitions_to_exit[emitting_state])

        logalpha = elnsum(logalpha, elnproduct(recur_alpha, recur_trans))

    elna_final = logalpha ### note -> this would actually be elnproduct(logalpha, eln(1)), but that is just logalpha

    # print(f'Forward algorithm final condition likelihood: {elna_final}') ## The log-likelihood!

    return elnaArr, elna_final

def log_backward(seqArr, A):

    transitions_from_initial = A[0, 1:-1]
    transitions_within_states = A[1:-1, 1:-1]
    transitions_to_exit = A[1:-1, -1]

    emitting_states = transitions_within_states.shape[0]

    density_initial, densities = seqArr[0], seqArr[1:]

    ### Initial Conditions ###
    # a) exit state (not in HTK book)
    # elnb_initial_exit = eln(1)  # for sure in the exit state at the end.
    # b) emitting states (1.23 in HTK Book)
    elnb_initial_emit = eln(transitions_to_exit)
    # c) entry state (not in HTK book)
    # elnb_initial_entry = eln(0) # can not be in entry state at the end
    elnbArr = np.array([elnb_initial_emit])

    ### Recursion ###
    for densArr in reversed(densities):
        last_elnb = elnbArr[-1]
        logbeta = np.array([LOGZERO] * emitting_states) ### initialize across emitting states

        for emitting_state in range(emitting_states):
            recur_beta = last_elnb[emitting_state]
            recur_trans = eln(transitions_within_states[:, emitting_state])
            recur_dens = eln(densArr[emitting_state])

            logbeta = elnsum(logbeta,
                             elnproduct(recur_trans,
                                        elnproduct(recur_dens,
                                                   recur_beta)))

        elnb = logbeta
        elnbArr = np.append(elnbArr, [elnb], axis = 0)

    ### Final Condition ###
    last_elnb = elnbArr[-1]
    logbeta = LOGZERO

    for emitting_state in range(emitting_states):
        recur_beta = last_elnb[emitting_state]
        recur_trans = eln(transitions_from_initial[emitting_state])
        recur_dens = eln(density_initial[emitting_state])

        logbeta = elnsum(logbeta,
                         elnproduct(recur_trans,
                                    elnproduct(recur_dens,
                                               recur_beta)))


    elnb_final = logbeta ### note -> this would actually be elnproduct(logalpha, eln(1)), but that is just logalpha

    # print(f'Backward algorithm final condition likelihood: {elnb_final}') ## The log-likelihood!

    elnbArr = np.flip(elnbArr, axis = 0) ### switch around time, since recursion was done in reverse.

    return elnbArr

def log_gamma(elnaArr, elnbArr):

    number_of_periods, emitting_states = elnaArr.shape

    ### Numerator
    eln_numerator = elnproduct(elnaArr, elnbArr)

    ### Denominator
    eln_denominator = np.array([LOGZERO] * number_of_periods)

    for emitting_state in range(emitting_states):
        eln_denominator = elnsum(eln_denominator, eln_numerator[:,emitting_state])

    ### Result
    eln_gamma = elnproduct(eln_numerator, -eln_denominator[:, np.newaxis])

    return eln_gamma


def log_xi(elnaArr, elnbArr, seqArr, A):

    number_of_periods, emitting_states = elnaArr.shape

    # transitions_from_initial = A[0, 1:-1]
    transitions_within_states = A[1:-1, 1:-1]
    # transition_to_exit = A[1:-1, -1]

    elna_without_last = elnaArr[:-1] ### shape: Time, S_from
    elnb_without_first = elnbArr[1:] ### shape: Time, S_to
    elndens_without_first = eln(seqArr[1:]) ### shape: Time, S_to
    elntrans_within_states = eln(transitions_within_states)

    ### pre-broadcast to shape: Time, S_from, S_to
    elna_without_last = elna_without_last[:, :, np.newaxis]
    elnb_without_first = elnb_without_first[:, np.newaxis, :]
    elndens_without_first = elndens_without_first[:, np.newaxis, :]
    elntrans_within_states = elntrans_within_states[np.newaxis, :, :]

    ### Numerator
    eln_numerator = elnproduct(elna_without_last,
                               elnproduct(elntrans_within_states,
                                          elnproduct(elndens_without_first,
                                                     elnb_without_first)))

    ### Denominator
    eln_denominator = np.array([LOGZERO] * (number_of_periods - 1))

    for from_state in range(emitting_states):
        for to_state in range(emitting_states):
            eln_denominator = elnsum(eln_denominator, eln_numerator[:, from_state, to_state])

    ### Result
    eln_xi = elnproduct(eln_numerator, -eln_denominator[:, np.newaxis, np.newaxis])

    return eln_xi

def weighted_emission_mean_numerator(elngammaArr, wordArr):
    return np.einsum('ts, tv -> sv', eexp(elngammaArr), wordArr)

def weighted_emission_cov_numerator(elngammaArr, wordArr, avgArr):
    # elngammaArr -> shape: T, S
    # wordArr -> shape: T, V
    # avgArr -> shape: S, V

    devArr = wordArr[:, np.newaxis, :] - avgArr[np.newaxis, :, :] ### shape -> Time x States x Variables
    covArr = np.einsum('ts, tsv, tsw -> svw', eexp(elngammaArr), devArr, devArr)
    return covArr

def weighted_denominator(elngammaArr):
    return eexp(elngammaArr).sum(axis = 0)

def weighted_transition_numerators(elnxiArr, elngammaArr):
    numerator_entry = eexp(elngammaArr[0])
    numerator_emit = eexp(elnxiArr).sum(axis = 0)
    numerator_exit = eexp(elngammaArr[-1])
    return numerator_entry, numerator_emit, numerator_exit

def join_entry_emit_and_exit_to_A(entryArr, emitArr, exitArr):
    # entryArr -> shape: S
    # emitArr -> shape: S, S
    # exitArr -> shape: S

    ### Helper function for prepending, analogous to appending
    def prepend(arr, values, axis = 0):
        return np.insert(arr, 0, values, axis = axis)

    def append(arr, values, axis = 0):
        ### since np append is very particular with object shapes and insert is not,
        ### we will leverage its generality.
        ### Problem with insert -> always before the index. Solution -> flip the array.
        flipArr = np.flip(arr, axis = axis)
        flipArr = prepend(flipArr, values, axis = axis)
        outArr = np.flip(flipArr, axis = axis)
        return outArr

    A = emitArr.copy()

    ### Add Entry State
    A = prepend(A, entryArr, axis = 0) ### add the transitions
                                        ## from state 0 to any emitting state

    A = prepend(A, 0, axis = 1) ### add the non-existent transitions
                                 ## from emitting states to zero
                                 ## i.e: in column 0 insert value 0.

    ### Add Exit State
    exitArr = prepend(exitArr, 0, axis = 0) ### add the non existent transition from entry to exit.

    A = append(A, exitArr, axis = 1) ### add the exit transitions

    A = append(A, 0, axis = 0) ### add the non-existent transitions away from the exit state.

    return A

def HRest(words,
          initial_A = None, initial_avg = None, initial_cov = None,
          stop_iteration_threshold = 10 ** -8,
          min_iterations = 15, max_iterations = 50,
          density_floor = 10 ** -100):

    from kenosha_hmm.utils import stop_condition, calculate_emission_densities

    if initial_A is None or initial_avg is None or initial_cov is None:
        from kenosha_hmm.HInit import HInit
        A, avg, cov = HInit(words)
    else:
        A, avg, cov = initial_A, initial_avg, initial_cov

    ####### GENERAL INITIALIZATION ######
    curr_log_likelihood = None
    prev_log_likelihood = None
    best_log_likelihood = None

    best_A, best_avg, best_cov = A, avg, cov

    num_iterations = 1

    while not stop_condition(curr_log_likelihood, prev_log_likelihood, num_iterations,
                             relative_change_threshold = stop_iteration_threshold,
                             min_iterations = min_iterations, max_iterations = max_iterations):

        ###### ITERATION SPECIFIC INITIALIZATION ######
        ### a) save previous log likelihood
        prev_log_likelihood = curr_log_likelihood

        ### b) observation specific quantities
        try:
            densFrm = calculate_emission_densities(words, avg, cov, density_floor = density_floor)
        except np.linalg.LinAlgError:
            lklStr = round(best_log_likelihood, 2) if best_log_likelihood is not None else None
            print(f'Best Likelihood: {lklStr}')
            return best_A, best_avg, best_cov

        obsIdxs = words.index.get_level_values(WORD_ID_COL).unique()
        avgArr = avg.values

        ### c) Accumulators
        ## Initialize Accumulators for gamma, which will feature in many denominators
        gammaAccumulator = 0

        ## Initialize Accumulators for mean and standard deviation numerators
        meanAccumulator = 0
        covAccumulator = 0

        ## Initialize Accumulators for xi, the numerator in transition probabilities
        xientryAccumulator = 0
        xiemitAccumulator = 0
        xiexitAccumulator = 0

        ## Initialize Accumulator for log-likelihood
        lklAccumulator = 0

        ###### UPDATE PARAMETERS ITERATION #####
        for obs in obsIdxs:
            seqArr = densFrm.loc[obs].values
            wordArr = words.loc[obs].values

            elnaArr, obs_log_likelihood = log_forward(seqArr, A)
            elnbArr = log_backward(seqArr, A)

            elngammaArr = log_gamma(elnaArr, elnbArr)
            elnxiArr = log_xi(elnaArr, elnbArr, seqArr, A)

            gammaAccumulator += weighted_denominator(elngammaArr)
            meanAccumulator += weighted_emission_mean_numerator(elngammaArr, wordArr)
            covAccumulator += weighted_emission_cov_numerator(elngammaArr, wordArr, avgArr)

            xi_entry, xi_emit, xi_exit = weighted_transition_numerators(elnxiArr, elngammaArr)
            xientryAccumulator += xi_entry
            xiemitAccumulator += xi_emit
            xiexitAccumulator += xi_exit

            lklAccumulator += obs_log_likelihood

        ####### CALCULATE ITERATION RESULTS #####
        upd_avgArr = np.einsum('iv, i -> iv', meanAccumulator, 1 / gammaAccumulator)
        upd_covArr = np.einsum('ivw, i -> ivw', covAccumulator, 1 / gammaAccumulator)
        upd_entryArr = np.einsum('i,  -> i', xientryAccumulator, 1 / len(obsIdxs))
        upd_emitArr = np.einsum('ij, i -> ij', xiemitAccumulator, 1 / gammaAccumulator)
        upd_exitArr = np.einsum('i, i -> i', xiexitAccumulator, 1 / gammaAccumulator)

        ### bring them into shape so we get our target objects ....
        # ... avg
        avg = pd.DataFrame(upd_avgArr, index = avg.index, columns = avg.columns)

        # ... cov
        states = cov.index.get_level_values('State').unique()
        cov = pd.concat([pd.DataFrame(upd_covArr[stateIdx],
                                      index = cov.loc[[stateVal]].index,
                                      columns = cov.columns) for stateIdx, stateVal in enumerate(states)])

        # ... A
        A = join_entry_emit_and_exit_to_A(upd_entryArr, upd_emitArr, upd_exitArr)

        # ... Likelihood
        curr_log_likelihood = lklAccumulator

        ##### CHECK IF ITERATION CAN END ####

        ### Save the parameters of the best iteration so far
        if best_log_likelihood is None:
            best_A, best_avg, best_cov = A, avg, cov
            best_log_likelihood = curr_log_likelihood
        else:
            if curr_log_likelihood > best_log_likelihood:
                best_A, best_avg, best_cov = A, avg, cov
                best_log_likelihood = curr_log_likelihood

        num_iterations += 1

    else:
        lklStr = round(best_log_likelihood, 2) if best_log_likelihood is not None else None
        print(f'Best Likelihood: {lklStr}')
        return best_A, best_avg, best_cov
