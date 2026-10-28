# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 22:35:57 2022

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

def join(*elems):
    if len(elems) == 0:
        return []

    x, xs = elems[0], elems[1:]

    try:
        scalarFlag = x.ndim == 0
    except AttributeError:
        x = np.array(x)
        scalarFlag = x.ndim == 0

    if scalarFlag:
        return [x[()], *join(*xs)]
    else:
        return [*x, *join(*xs)]


def log_forward(seqArr, A, prev_model_exit_array = None):
    ### ALGORITHM 5 in Tobias P. Mann

    transitions_from_initial = A[0, 1:-1]
    transitions_within_states = A[1:-1, 1:-1]
    transitions_to_exit = A[1:-1, -1]

    emitting_states = transitions_within_states.shape[0]

    density_initial, densities = seqArr[0], seqArr[1:]

    ### deal with injected previous model exit probabilities
    if prev_model_exit_array is None: # if we're in the first model, the probability of exiting
                                      # an imaginary previous model is 1 in the first period and 0 later.
        initial_entry_alpha, prev_model_exit_alphas = eln(1), [eln(0)] * len(densities)
    else: # if we're not in the first model,
            # the probability of exiting the previous model in the previous period
            # corresponds to the exit array without the last element
                # (note: prev_model_exit_alphas is of equal length of densities, but starts at index 0 whereas densities starts at index 1)
            # the probability of entering a non-initial model in period 0 is 0 -> no tee models allowed.
        initial_entry_alpha, prev_model_exit_alphas = eln(0), prev_model_exit_array[:-1]

    ### Initial Conditions ###
    # a) entry state
    elna_initial_entry = initial_entry_alpha
    # b) emitting states
    elna_initial_emit = elnproduct(eln(transitions_from_initial), eln(density_initial)) ### erroneous (??) equation in HTKBook, p.170
    elna_initial_emit = elnproduct(elna_initial_entry, elna_initial_emit) ### corrected!
    # c) exit state
    elna_initial_exit = LOGZERO
    for emitting_state in range(emitting_states):
        recur_alpha = elna_initial_emit[emitting_state]
        recur_trans = eln(transitions_to_exit[emitting_state])

        elna_initial_exit = elnsum(elna_initial_exit,
                                   elnproduct(recur_alpha,
                                              recur_trans))

    elna_initial = join(elna_initial_entry, elna_initial_emit, elna_initial_exit)
    elnaArr = np.array([elna_initial])

    ### Recursion ###
    # note that the recursion is done vectorized for all three states.
    # hence, we have one nested fewer loops (2 instead of 3) in the recursion compared to Mann (2006)
    for densArr, prev_model_exit_alpha in zip(densities, prev_model_exit_alphas):

        ### get last periods alphas
        last_elna = elnaArr[-1]
        _, last_elna_emit, _ = last_elna[0], last_elna[1:-1], last_elna[-1]

        ### entry state -> models can also be in entry state at time points later than t = 0
        elna_entry = prev_model_exit_alpha # here, we inject the prior-model-exit-log-alpha

        ### emitting state
        logalpha = elnproduct(eln(transitions_from_initial), elna_entry)  ### initialize across emitting states

        for emitting_state in range(emitting_states):
            recur_alpha = last_elna_emit[emitting_state]
            recur_trans = eln(transitions_within_states[emitting_state])

            logalpha = elnsum(logalpha,
                              elnproduct(recur_alpha,
                                         recur_trans))

        elna_emit = elnproduct(logalpha, eln(densArr))

        ### exit state -> models can be in exit state at time points earlier than t = T
        elna_exit = LOGZERO
        for emitting_state in range(emitting_states):
            recur_alpha = elna_emit[emitting_state]
            recur_trans = eln(transitions_to_exit[emitting_state])

            elna_exit = elnsum(elna_exit,
                               elnproduct(recur_alpha,
                                          recur_trans))

        ### Join all three arrays
        elna = join(elna_entry, elna_emit, elna_exit)
        elnaArr = np.append(elnaArr, [elna], axis = 0)

    # print(f'Forward algorithm final condition likelihood: {elna_final}') ## The log-likelihood!

    return elnaArr


def log_backward(seqArr, A, succ_model_entry_array = None):

    transitions_from_initial = A[0, 1:-1]
    transitions_within_states = A[1:-1, 1:-1]
    transitions_to_exit = A[1:-1, -1]

    emitting_states = transitions_within_states.shape[0]

    # density_initial, densities = seqArr[0], seqArr[1:]
    densities, density_final = seqArr[:-1], seqArr[-1]  #these are b(o_t) and b(o_T)
    densities_next = seqArr[1:] #these are b(o_(t+1))

    ### deal with injected previous model exit probabilities
    if succ_model_entry_array is None: # if we're in the last model, the probability of exiting
                                        # is 1 in the last period and 0 everywhere else
        succ_model_entry_betas, final_exit_beta = [eln(0)] * len(densities), eln(1)
    else: # if we're not in the last model,
            # the probability of entering the next model in the next period
            # corresponds to the exit array without the last element
                # (note: prev_model_exit_alphas is of equal length of densities, but starts at index 0 whereas densities starts at index 1)
            # the probability of exiting a non-final model in period T is 0 -> no tee models allowed.
        succ_model_entry_betas, final_exit_beta = succ_model_entry_array[1:], eln(0)

    ### Initial Conditions ###
    # a) exit state (not in HTK book)
    elnb_initial_exit = final_exit_beta  # for sure in the exit state at the end.
    # b) emitting states (1.23 in HTK Book)
    elnb_initial_emit = elnproduct(eln(transitions_to_exit), elnb_initial_exit)
    # c) entry state (not in HTK book)
    elnb_initial_entry = LOGZERO

    for emitting_state in range(emitting_states):
        recur_beta = elnb_initial_emit[emitting_state]
        recur_trans = eln(transitions_from_initial[emitting_state])
        recur_dens = eln(density_final[emitting_state])

        elnb_initial_entry = elnsum(elnb_initial_entry,
                                    elnproduct(recur_trans,
                                               elnproduct(recur_dens,
                                                          recur_beta)))

    elnb_initial = join(elnb_initial_entry, elnb_initial_emit, elnb_initial_exit)

    elnbArr = np.array([elnb_initial])

    ### Recursion ###
    for densArr_this, densArr_next, succ_model_entry_beta in zip(reversed(densities),
                                                                 reversed(densities_next),
                                                                 reversed(succ_model_entry_betas)):

        ### get last periods alphas
        last_elnb = elnbArr[-1]
        _, last_elnb_emit, _ = last_elnb[0], last_elnb[1:-1], last_elnb[-1]

        ### exit state -> models can also be in exit state at time points earlier than t = T
        elnb_exit = succ_model_entry_beta # here, we inject the prior-model-exit-log-alpha

        ### emitting state
        logbeta = elnproduct(eln(transitions_to_exit), elnb_exit)  ### initialize across emitting states

        for emitting_state in range(emitting_states):
            recur_beta = last_elnb_emit[emitting_state]
            recur_trans = eln(transitions_within_states[:, emitting_state])
            recur_dens = eln(densArr_next[emitting_state])

            logbeta = elnsum(logbeta,
                             elnproduct(recur_trans,
                                        elnproduct(recur_dens,
                                                   recur_beta)))

        elnb_emit = logbeta

        ### entry state
        elnb_entry = LOGZERO

        for emitting_state in range(emitting_states):
            recur_beta = elnb_emit[emitting_state]
            recur_trans = eln(transitions_from_initial[emitting_state])
            recur_dens = eln(densArr_this[emitting_state])

            elnb_entry = elnsum(elnb_entry,
                                elnproduct(recur_trans,
                                           elnproduct(recur_dens,
                                                      recur_beta)))

        ### Join all three arrays
        elnb = join(elnb_entry, elnb_emit, elnb_exit)
        elnbArr = np.append(elnbArr, [elnb], axis = 0)

    elnbArr = np.flip(elnbArr, axis = 0) ### switch around time, since recursion was done in reverse.

    return elnbArr

def get_sequence_data_and_transcription(embedded_training_sample, sequence_idx):
    smplFrm = embedded_training_sample[embedded_training_sample.IndexSF == sequence_idx]

    words = smplFrm.drop(columns = 'Event').reset_index().set_index(['IndexSF', 'Frame'])

    ### indexify models -> get order of models in sequence, without frame boundaries.
    eventList = smplFrm.Event.values

    def sequence_transcription(eventList):
        ###
        x, xs = eventList[0], eventList[1:]
        outList = [x]
        for x in xs:
            last = outList[-1]
            if x != last:
                outList.append(x)
        return outList

    eventSeq = sequence_transcription(eventList)

    return words, eventSeq


def get_sequence_densities(restDict, eventSeq, words, density_floor = 10 ** -100):
    from kenosha_hmm.utils import calculate_emission_densities

    ### 1) Calculate Emission Densities for each word-model
    seqDict = {}
    for word in set(eventSeq):

        HMM = restDict[word]
        _, avg, cov = HMM['A'], HMM['avg'], HMM['cov']

        densFrm = calculate_emission_densities(words, avg, cov, density_floor = density_floor)
        seqArr = densFrm.values

        seqDict[word] = seqArr

    return seqDict

def get_forward_probs(restDict, eventSeq, seqDict):

    ### 2) Calculate the forward probabilities
    prev_model_exit_array = None
    elnas = []
    for word in eventSeq:
        seqArr = seqDict[word]

        HMM = restDict[word]
        A, _, _ = HMM['A'], HMM['avg'], HMM['cov']

        elnaArr = log_forward(seqArr, A, prev_model_exit_array = prev_model_exit_array)
        prev_model_exit_array = elnaArr[:, -1]

        elnas.append(elnaArr)

    return elnas

def get_backward_probs(restDict, eventSeq, seqDict):

    ### 3) Calculate the backwards probabilities
    succ_model_entry_array = None
    elnbs = []
    for word in reversed(eventSeq):
        seqArr = seqDict[word]

        HMM = restDict[word]
        A, _, _ = HMM['A'], HMM['avg'], HMM['cov']

        elnbArr = log_backward(seqArr, A, succ_model_entry_array = succ_model_entry_array)
        succ_model_entry_array = elnbArr[:, 0]

        elnbs.append(elnbArr)

    elnbs = list(reversed(elnbs))

    return elnbs

def log_gammas_and_model_prob(elnas, elnbs):

    number_of_periods = elnas[0].shape[0]

    eln_numerators = [elnproduct(elnaArr, elnbArr) for elnaArr, elnbArr in zip(elnas, elnbs)]

    def calc_denominator(eln_numerator):
        number_of_periods, emitting_states = eln_numerator.shape
        emitting_states -= 2 ### because elnaArr also contains entry and exit.

        ### Denominator
        eln_denominator = np.array([LOGZERO] * number_of_periods)

        for emitting_state in range(1, emitting_states + 1):
            eln_denominator = elnsum(eln_denominator, eln_numerator[:,emitting_state])

        return eln_denominator

    eln_denominators = [calc_denominator(eln_numerator) for eln_numerator in eln_numerators]

    overall_denominator = np.array([LOGZERO] * number_of_periods)
    for eln_denominator in eln_denominators:
        overall_denominator = elnsum(overall_denominator, eln_denominator)

    ### check the likelihood
    # 1) Using final forward and backward conditions
    mod_prob_alpha = elnas[-1][-1][-1]
    mod_prob_beta = elnbs[0][0][0]

    assert np.isclose(mod_prob_alpha, mod_prob_beta), f'Different Likelihoods - Alpha {mod_prob_alpha}, Beta: {mod_prob_beta}'

    # 2) Using denominator
    denom_mean = overall_denominator.mean()

    assert all(np.isclose(overall_denominator, denom_mean)), 'Likelihood via gamma sum inconsistent!'

    # 3) combining both
    assert all(np.isclose([mod_prob_alpha, mod_prob_beta], denom_mean)), 'Final conditions inconsistent with gamma sum'

    # 3) yield likelihood
    model_probability = denom_mean

    ### Result
    eln_gammas = [elnproduct(eln_numerator, -model_probability) for eln_numerator in eln_numerators]

    return eln_gammas, model_probability


def log_xi_emit_numerator(elnaArr, elnbArr, seqArr, A):

    elnaArr_emit = elnaArr[:, 1:-1]
    elnbArr_emit = elnbArr[:, 1:-1]

    number_of_periods, emitting_states = elnaArr.shape

    # transitions_from_initial = A[0, 1:-1]
    transitions_within_states = A[1:-1, 1:-1]
    # transition_to_exit = A[1:-1, -1]

    elna_without_last = elnaArr_emit[:-1] ### shape: Time, S_from
    elnb_without_first = elnbArr_emit[1:] ### shape: Time, S_to
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

    return eln_numerator

def log_xi_entry_numerator(elnaArr, elnbArr, seqArr, A):

    number_of_periods, _ = elnaArr.shape

    transitions_from_initial = A[0, 1:-1]
    # transitions_within_states = A[1:-1, 1:-1]
    # transition_to_exit = A[1:-1, -1]

    elnaArr_entry = elnaArr[:, 0] # Include entry on the final frame
    elnbArr_emit = elnbArr[:, 1:-1]
    elndens = eln(seqArr) # Entry and emission occur on the same frame
    elntrans_from_initial = eln(transitions_from_initial)

    ### pre-broadcast to shape: Time, S_to
    elnaArr_entry = elnaArr_entry[:, np.newaxis]
    # elnbArr_emit = elnbArr_emit
    # elndens = elndens
    elntrans_from_initial = elntrans_from_initial[np.newaxis, :]

    eln_numerator = elnproduct(elnaArr_entry,
                               elnproduct(elntrans_from_initial,
                                          elnproduct(elndens, elnbArr_emit)))

    return eln_numerator

def log_xi_exit_numerator(elnaArr, elnbArr, seqArr, A):

    number_of_periods, _ = elnaArr.shape

    # transitions_from_initial = A[0, 1:-1]
    # transitions_within_states = A[1:-1, 1:-1]
    transition_to_exit = A[1:-1, -1]

    elnaArr_emit = elnaArr[:, 1:-1] # Include the final model's terminal exit
    elnbArr_exit = elnbArr[:, -1]
    elntrans_to_exit = eln(transition_to_exit)

    ### pre-broadcast to shape: Time, S_from
    elnbArr_exit = elnbArr_exit[:, np.newaxis]
    elntrans_to_exit = elntrans_to_exit[np.newaxis, :]

    eln_numerator = elnproduct(elnaArr_emit,
                               elnproduct(elntrans_to_exit,
                                          elnbArr_exit))
    return eln_numerator

def HERest(init_hmmDict, dataFrm, num_cycles = 4, tikhonov_regularizer = 0):

    ### Import from HRest
    from kenosha_hmm.HRest import weighted_denominator
    from kenosha_hmm.HRest import weighted_emission_cov_numerator, weighted_emission_mean_numerator
    from kenosha_hmm.HRest import join_entry_emit_and_exit_to_A

    ### Preparation before the Baum Welch Cycles
    seqIdxs = dataFrm.IndexSF.unique()
    print(f'Available Sequences: {len(seqIdxs)}')
    all_words = dataFrm.Event.unique()

    from copy import deepcopy
    paramDict = deepcopy(init_hmmDict)

    ### Repeat Embedded Baum Welch, num_cycles times.
    for cycleNum in range(1, num_cycles + 1):
        ### Initialize Accumulators
        ## Initialize Accumulators for gamma, which will feature in many denominators
        gammaAccumulator = {word: 0 for word in all_words}
        gammaAccumulator_entry = {word: 0 for word in all_words}

        ## Initialize Accumulators for mean and standard deviation numerators
        meanAccumulator = {word: 0 for word in all_words}
        covAccumulator = {word: 0 for word in all_words}

        ## Initialize Accumulators for xi, the numerator in transition probabilities
        xientryAccumulator = {word: 0 for word in all_words}
        xiemitAccumulator = {word: 0 for word in all_words}
        xiexitAccumulator = {word: 0 for word in all_words}

        ## Initialize Accumulator for log-likelihood
        lklAccumulator = 0

        for i, sequenceIdx in enumerate(seqIdxs):

            if i % 10 == 0:
                print(f'Embedded re-estimation of sequence {i} out of {len(seqIdxs)}...')

            words, eventSeq = get_sequence_data_and_transcription(dataFrm, sequenceIdx)

            seqDict = get_sequence_densities(paramDict, eventSeq, words)

            elnas = get_forward_probs(paramDict, eventSeq, seqDict)
            elnbs = get_backward_probs(paramDict, eventSeq, seqDict)

            eln_gammas, model_probability = log_gammas_and_model_prob(elnas, elnbs)

            for idx, word in enumerate(eventSeq):

                elnaArr, elnbArr, seqArr = elnas[idx], elnbs[idx], seqDict[word]
                HMM = paramDict[word]
                A, avg, cov = HMM['A'], HMM['avg'], HMM['cov']
                avgArr = avg.values
                features = list(avg.columns)
                wordArr = words[features].values

                ### calculate emission accumulators
                elngammaArr = eln_gammas[idx][:, 1:-1] ### gammas for this index, for emitting states
                gammaAccumulator[word] += weighted_denominator(elngammaArr)
                meanAccumulator[word] += weighted_emission_mean_numerator(elngammaArr, wordArr)
                covAccumulator[word] += weighted_emission_cov_numerator(elngammaArr, wordArr, avgArr)

                ### calculate gamma accumulators for entry states (to normalize accumulator for entry xi)
                elngammaArr_entry = eln_gammas[idx][:, 0]
                gammaAccumulator_entry[word] += weighted_denominator(elngammaArr_entry)

                ### calculate xi numerators
                log_xi_entry_num = log_xi_entry_numerator(elnaArr, elnbArr, seqArr, A)
                log_xi_emit_num = log_xi_emit_numerator(elnaArr, elnbArr, seqArr, A)
                log_xi_exit_num = log_xi_exit_numerator(elnaArr, elnbArr, seqArr, A)

                ### divide by model probability
                log_xi_entry = elnproduct(log_xi_entry_num, -model_probability)
                log_xi_emit = elnproduct(log_xi_emit_num, -model_probability)
                log_xi_exit = elnproduct(log_xi_exit_num, -model_probability)

                ### calculate transition accumulators
                transition_entry_numerator = eexp(log_xi_entry).sum(axis = 0)
                transition_emit_numerator = eexp(log_xi_emit).sum(axis = 0)
                transition_exit_numerator = eexp(log_xi_exit).sum(axis = 0)

                xientryAccumulator[word] += transition_entry_numerator
                xiemitAccumulator[word] += transition_emit_numerator
                xiexitAccumulator[word] += transition_exit_numerator

            ### accumulate likelihood
            lklAccumulator = elnproduct(lklAccumulator, model_probability)


        print(f'Embedded re-estimation cycle {cycleNum} done! Total Likelihood: {lklAccumulator}')

        for word in all_words:

            ####### CALCULATE ITERATION RESULTS #####
            upd_avgArr = np.einsum('iv, i -> iv', meanAccumulator[word], 1 / gammaAccumulator[word])
            upd_covArr = np.einsum('ivw, i -> ivw', covAccumulator[word], 1 / gammaAccumulator[word])
            upd_entryArr = np.einsum('i,  -> i', xientryAccumulator[word], 1 / gammaAccumulator_entry[word])
            upd_emitArr = np.einsum('ij, i -> ij', xiemitAccumulator[word], 1 / gammaAccumulator[word])
            upd_exitArr = np.einsum('i, i -> i', xiexitAccumulator[word], 1 / gammaAccumulator[word])

            ##### TIKHONOV REGULARIZATION #######
            regArr = np.eye(*upd_covArr.shape[1:])[np.newaxis, :, :] * tikhonov_regularizer
            upd_covArr = upd_covArr + regArr

            ##### UPDATE THE HMM PARAMETERS #####
            HMM = paramDict[word]
            A, avg, cov = HMM['A'], HMM['avg'], HMM['cov']

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

            paramDict[word] = {'A': A, 'avg': avg, 'cov': cov}

    return paramDict
