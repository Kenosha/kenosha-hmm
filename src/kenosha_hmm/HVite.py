# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 02:29:18 2022

@author: gianl
"""

import pandas as pd
import numpy as np

from kenosha_hmm.utils import calculate_emission_densities

@np.errstate(divide='ignore')
def token_pass_single_word(hmmDict, word, density_floor = 10 ** -100):

    lklDict = {}

    for wordStr, hmm in hmmDict.items():
        A, avg, cov = hmm['A'], hmm['avg'], hmm['cov']

        logA = np.log(A)
        log_transitions_from_initial = logA[0, 1:-1]
        log_transitions_within_states = logA[1:-1, 1:-1]
        log_transition_to_exit = logA[1:-1, -1]

        densFrm = calculate_emission_densities(word, avg, cov, density_floor = density_floor)
        log_densArr = np.log(densFrm.values)
        log_density_initial, log_densities = log_densArr[0], log_densArr[1:]

        num_states = len(avg)

        ### Initialize Token Array
        log_token_values = log_transitions_from_initial + log_density_initial

        ### Recurse Token Array across emitting states
        for log_dens_values in log_densities:

            ### broadcast these to SxS
            log_token_values_brd = log_token_values[:, np.newaxis] #### shape: S_from, S_to
            log_dens_values_brd = log_dens_values[np.newaxis, :] ### shape: S_from, S_to
            log_transitions_within_states ### shape: S_from, S_to

            new_token_values_brd = log_token_values_brd + log_transitions_within_states + log_dens_values_brd

            log_token_values = new_token_values_brd.max(axis = 0) ### in every column, find maximum

        ### Model Transition into Exit State
        log_token_values = log_token_values + log_transition_to_exit

        log_token_final = log_token_values.max()

        lklDict[wordStr] = log_token_final

    return pd.Series(lklDict).sort_values(ascending = False)


@np.errstate(divide='ignore')
def token_pass_sequence(hmmDict, words, density_floor = 10 ** -100):

    ### Get some HMM-specific data
    D = {}
    for wordStr, hmm in hmmDict.items():
        A, avg, cov = hmm['A'], hmm['avg'], hmm['cov']

        logA = np.log(A)
        log_transitions_from_initial = logA[0, 1:-1]
        log_transitions_within_states = logA[1:-1, 1:-1]
        log_transition_to_exit = logA[1:-1, -1]

        densFrm = calculate_emission_densities(words, avg, cov, density_floor = density_floor)
        log_densArr = np.log(densFrm.values)
        # log_density_initial, log_densities = log_densArr[0], log_densArr[1:]

        D[wordStr] = {'log_A_init': log_transitions_from_initial,
                      'log_A_emit': log_transitions_within_states,
                      'log_A_exit': log_transition_to_exit,
                      'log_dens': log_densArr,
                      'states': len(avg)}

    ### Initialize a list of WLRs
    WLRs = []

    ### Initialize token values
    for wordStr in hmmDict:

        D[wordStr]['log_token_values'] = np.log(np.zeros(D[wordStr]['states']))
        D[wordStr]['path_identifier'] = np.ones(D[wordStr]['states'], dtype = int) * -1


    ### Recurse Token Array and path identifiers
    for t in range(len(words)):

        ### CREATE WLR FOR BEST TOKEN EXIT VALUE FROM PREVIOUS MODEL ###
        WLR = {'id': t, 'period_exited': t - 1}
        if t == 0:
            WLR['emitting_model'] = None
            WLR['token_value'] = np.log(1)
            WLR['token_path_identifier'] = None
        else:
            for wordStr in hmmDict:
                exit_value = D[wordStr]['exit_token_value']
                exit_path = D[wordStr]['exit_path_identifier']

                try:
                    WLR['token_value']
                except:
                    WLR['emitting_model'] = wordStr
                    WLR['token_value'] = exit_value
                    WLR['token_path_identifier'] = exit_path

                else:
                    if exit_value > WLR['token_value']:
                        WLR['emitting_model'] = wordStr
                        WLR['token_value'] = exit_value
                        WLR['token_path_identifier'] = exit_path

        WLRs.append(WLR)

        for wordStr in hmmDict:

            ####### TRANSITION INTO HMM ###########
            ### get relevant values
            best_entry_token_value = WLR['token_value']
            log_transitions_from_entry = D[wordStr]['log_A_init']
            log_dens_values = D[wordStr]['log_dens'][t]

            entry_token_values = best_entry_token_value + log_transitions_from_entry
            entry_path_identifiers = np.ones(D[wordStr]['states'], dtype = int) * WLR['id']

            ### COMPARE TRANSITION INTO WITH WHAT IS ALREADY WITHIN
            emit_token_values = D[wordStr]['log_token_values']
            emit_path_identifiers = D[wordStr]['path_identifier']
            entry_likelier = entry_token_values > emit_token_values ### by state, is it likelier
                                                                    ### to come from outside?
            new_token_values = np.where(entry_likelier, entry_token_values, emit_token_values)
            new_path_identifiers = np.where(entry_likelier, entry_path_identifiers, emit_path_identifiers)

            ### EMIT OBSERVATIONS (and add log_density to likelihood) ###
            D[wordStr]['log_token_values'] = new_token_values + log_dens_values
            D[wordStr]['path_identifier'] = new_path_identifiers

            ####### TRANSITION WITHIN HMM #########
            ### get relevant values
            log_token_values = D[wordStr]['log_token_values']
            log_transitions_within_states = D[wordStr]['log_A_emit']

            ### broadcast these to SxS
            log_token_values_brd = log_token_values[:, np.newaxis] #### shape: S_from, S_to
            log_transitions_within_states ### shape: S_from, S_to

            ### calculate token values
            emit_token_values_brd = log_token_values_brd + log_transitions_within_states
            emit_token_values = emit_token_values_brd.max(axis = 0) ### in every column, find maximum
            emit_token_indices = emit_token_values_brd.argmax(axis = 0)
            emit_path_identifiers = D[wordStr]['path_identifier'][emit_token_indices]

            ### Update the hmm data
            D[wordStr]['log_token_values'] = emit_token_values
            D[wordStr]['path_identifier'] = emit_path_identifiers

            ###### TRANSITION OUT OF HMM ###########
            ### get relevant values
            log_transitions_to_exit = D[wordStr]['log_A_exit']

            ### calculate token values
            exit_token_values = emit_token_values + log_transitions_to_exit
            exit_token_value = exit_token_values.max(axis = 0) ### in every column, find maximum
            exit_token_index = exit_token_values.argmax(axis = 0)

            ### Update the hmm data
            D[wordStr]['exit_token_value'] = exit_token_value
            D[wordStr]['exit_path_identifier'] = D[wordStr]['path_identifier'][exit_token_index]


        ### CHECK HIGHEST EXIT TOKEN IN LAST PERIOD
        max_exit_value = None
        for wordStr in hmmDict:
            curr_exit_value = D[wordStr]['exit_token_value']
            if max_exit_value is None:
                max_exit_value = curr_exit_value
                max_exit_model = wordStr
            elif max_exit_value < curr_exit_value:
                max_exit_value = curr_exit_value
                max_exit_model = wordStr

    ### RECONSTRUCT LINKED WORD LIST
    sequence_frames = words.index

    inferred_word_sequence = [] ### list of tuples -> frame_when_word_was_left, word

    inferred_word_sequence.append((sequence_frames[t], max_exit_model))
    last_id = D[max_exit_model]['exit_path_identifier']

    while not last_id == 0:
        WLR = WLRs[last_id]
        max_exit_model = WLR['emitting_model']
        period_exited = WLR['period_exited']
        inferred_word_sequence.append((sequence_frames[period_exited], max_exit_model))

        last_id = WLR['token_path_identifier']

    inferred_word_sequence = list(reversed(inferred_word_sequence))

    return inferred_word_sequence
