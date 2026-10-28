# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 00:46:42 2022

@author: gianl
"""


def init_all_single_mixture(word_training_sample,
                            featDict = None,
                            event_to_num_statesFunc = None,
                            train_threshold = 8,
                            stop_iteration_threshold_init = 10 ** -10,
                            stop_iteration_threshold_rest = 10 ** -7,
                            min_iterations = 15, max_iterations = 50,
                            density_floor = 10 ** -100):

    all_words = word_training_sample.index.get_level_values('Event').unique()
    print(f'Initializing the following words: {all_words}')

    if event_to_num_statesFunc is None:
        event_to_num_statesFunc = lambda eventStr: 3 ### have three events regardless of word

    initDict = {}
    restDict = {}
    for wordStr in all_words:

        ### get number of states
        num_states = event_to_num_statesFunc(wordStr)

        ### get features
        if featDict is None:
            features = word_training_sample.columns
        else:
            features = featDict[wordStr]

        print(f'Doing {wordStr}...')

        words = word_training_sample.loc[wordStr, features]
        num_words = len(words.index.get_level_values("IndexWord").unique())
        num_frames = len(words)
        print(f'Number of words to train on: {num_words}, occupying {num_frames} frames')

        from kenosha_hmm.HInit import HInit
        from kenosha_hmm.HRest import HRest

        if num_words < train_threshold:  ### in this case, we initialize the parameters
                                            # with even segmentation and global variance
            A, avg, cov = HInit(words, num_states = num_states,
                                min_iterations = 0, max_iterations = 1, cov_global = True)
            ### this is one iteration of HInit, i.e, a even-segmentation initialization.
            initDict[wordStr] = {'A': A, 'avg': avg, 'cov': cov}
            restDict[wordStr] = {'A': A, 'avg': avg, 'cov': cov}

        else:
            initial_A, initial_avg, initial_cov = HInit(words, num_states = num_states,
                                                        stop_iteration_threshold = stop_iteration_threshold_init,
                                                        min_iterations = min_iterations,
                                                        max_iterations = max_iterations,
                                                        density_floor = density_floor)

            initDict[wordStr] = {'A': initial_A, 'avg': initial_avg, 'cov': initial_cov}
            print('')

            A, avg, cov = HRest(words,
                                initial_A = initial_A, initial_avg = initial_avg, initial_cov = initial_cov,
                                stop_iteration_threshold = stop_iteration_threshold_rest,
                                min_iterations = min_iterations, max_iterations = max_iterations,
                                density_floor = density_floor)

            restDict[wordStr] = {'A': A, 'avg': avg, 'cov': cov}

            print('... Done!')
            print('______________')

    return restDict
