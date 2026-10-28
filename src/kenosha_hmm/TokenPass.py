# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 00:19:40 2022

@author: gianl
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass

from kenosha_hmm.utils import calculate_emission_densities


def flatten(someList):
    return [el for inList in someList for el in inList]

@dataclass
class HMM:
    A: np.array
    avg: pd.DataFrame
    cov: pd.DataFrame

    def __post_init__(self):
        self.log_A = np.log(self.A)
        self.log_A_init = self.log_A[0, 1:-1]
        self.log_A_emit = self.log_A[1:-1, 1:-1]
        self.log_A_exit = self.log_A[1:-1, -1]
        self.states = len(self.avg)

        self.isTerminal = True ### word models are terminal symbols in our syntax!

        self.entry_token_value = np.log(1)
        self.entry_token_id = -1

        self.token_values = np.log(np.zeros(self.states))
        self.token_ids = np.ones(self.states, dtype = int) * -1

        self.exit_token_value = np.log(0)
        self.exit_token_id = -1

    def init_log_density(self, words, density_floor = 10 ** -100):
        densFrm = calculate_emission_densities(words, self.avg, self.cov,
                                               density_floor = density_floor)
        self.log_densFrm = np.log(densFrm)

    def get_density_at(self, t):
        return self.log_densFrm.loc[t].values

    def set_entry_token(self, token_value, token_id):
        self.entry_token_value = token_value
        self.entry_token_id = token_id

    def get_exit_token(self):
        return self.exit_token_value, self.exit_token_id

    def step_tokens_into_hmm(self, t):
        ### token value: scalar
        ### token id: scalar

        ####### TRANSITION INTO HMM ###########
        entry_token_values = self.entry_token_value + self.log_A_init
        entry_token_ids = [self.entry_token_id] * self.states

        return entry_token_values, entry_token_ids

    def step_tokens_within_hmm(self, t):

        ####### TRANSITION WITHIN HMM #########

        ### broadcast these to SxS
        token_values_brd = self.token_values[:, np.newaxis] #### shape: S_from, S_to

        ### calculate token values
        emit_token_values_brd = token_values_brd + self.log_A_emit
        emit_token_values = emit_token_values_brd.max(axis = 0) ### in every column, find maximum
        emit_token_indices = emit_token_values_brd.argmax(axis = 0)
        emit_token_ids = self.token_ids[emit_token_indices]

        return emit_token_values, emit_token_ids

    def step_tokens_out_of_hmm(self):
        ###### TRANSITION OUT OF HMM ###########

        ### calculate token values
        exit_token_values = self.token_values + self.log_A_exit
        exit_token_value = exit_token_values.max(axis = 0) ### in every column, find maximum
        exit_token_index = exit_token_values.argmax(axis = 0)
        exit_token_ids = self.token_ids[exit_token_index]

        return exit_token_value, exit_token_ids

    def set_step_tokens(self, t):
        ### t - epsilon: COMPARE TRANSITION INTO WITH WHAT IS ALREADY WITHIN
        entry_token_values, entry_token_ids = self.step_tokens_into_hmm(t)
        emit_token_values, emit_token_ids = self.token_values, self.token_ids

        entry_likelier = entry_token_values > emit_token_values ### by state, is it likelier
                                                                ### to come from outside?

        new_token_values = np.where(entry_likelier, entry_token_values, emit_token_values)
        new_token_ids = np.where(entry_likelier, entry_token_ids, emit_token_ids)

        ### t: EMIT OBSERVATIONS (and add log_density to likelihood)
        self.token_values = new_token_values + self.get_density_at(t)
        self.token_ids = new_token_ids

        ### t + epsilon: STEP TOKENS FROM WITHIN HMM
        # a) into emitting states
        emit_token_values, emit_token_ids = self.step_tokens_within_hmm(t)

        # b) into exit
        exit_token_value, exit_token_ids = self.step_tokens_out_of_hmm()

        ### Re-set all token values after t
        self.token_values = emit_token_values
        self.token_ids = emit_token_ids

        self.exit_token_value = exit_token_value
        self.exit_token_id = exit_token_ids

        self.entry_token_value = np.log(0)
        self.entry_token_id = -1



@dataclass
class Link:
    prec: list ### list of preceeding models
    succ: list ### list of succeeding models

    def __post_init__(self):
        self.token_value = np.log(0)
        self.token_id = -1
        self.emitting_model = None

    def set_filter_tokens(self, model, model_name):
        model_exit_token_value, model_exit_token_id = model.get_exit_token()
        if model_exit_token_value > self.token_value:
            self.token_value = model_exit_token_value
            self.token_id = model_exit_token_id
            self.emitting_model = model_name

    def record_decisions(self, t, WLRs):
            # Shared grammar links can propagate the same terminal exit twice
            # on one frame. Reuse its existing record rather than inventing a
            # second, zero-duration event in the backtrace.
            if isinstance(self.token_id, str):
                previous_t, previous_num = map(int, self.token_id.split('_'))
                previous = WLRs[previous_t][previous_num]
                if (previous['t'] == t
                        and previous['emitting_model'] == self.emitting_model
                        and previous['token_value'] == self.token_value):
                    return

            ### make new wlr entry
            wlrs_at_t = WLRs[t]
            if wlrs_at_t == {}:
                new_key = 0
            else:
                new_key = max(wlrs_at_t) + 1 ### get the maximum key, add one

            new_wlr_entry = {'token_value': self.token_value,
                             'token_id': self.token_id,
                             't': t,
                             'emitting_model': self.emitting_model}
            wlrs_at_t[new_key] = new_wlr_entry

            ### replace token id with wlr pointer
            self.token_id = str(t) + '_' + str(new_key)


    def push_token_to_entry(self, t):
        for shell in self.succ:
            shell.inject_token(self.token_value, self.token_id)
            model = shell.model
            if not model is None: ### only if the model shell has been activated.
                if not model.isTerminal:
                    model.propagate_entry_tokens(t)
                else:
                    model.set_step_tokens(t) ### when pushing, we also move ahead terminal models

        self.token_value = np.log(0)
        self.token_id = -1

    def pull_token_from_exit(self, t, WLRs):
        for shell in self.prec:
            model, name = shell.model, shell.name
            if not model is None: ### only if the model shell has been activated.
                if not model.isTerminal:
                    model.propagate_exit_tokens(t, WLRs)
                self.set_filter_tokens(model, name)
        if self.emitting_model is not None: ### if the link has been reached with positive prob
            self.record_decisions(t, WLRs)

@dataclass
class BinaryLink:
    ### a link connecting one model to another model
    prec: None # ModelShell
    succ: None # ModelShell

    def reverse(self):
        return BinaryLink(self.succ, self.prec)

    def is_entry(self):
        return self.prec is None
    def is_exit(self):
        return self.succ is None

@dataclass
class HalfLink:
    ### a link connecting one model with many other models
    prec: None # ModelShell
    succ: list


@dataclass
class SubSyntaxModel:
    entry_link: Link
    intm_links: list
    exit_link: Link

    def __post_init__(self):
        self.isTerminal = False ### sub syntax models consist of other sub syntax models or HMMs

    def set_entry_token(self, token_value, token_id):
        self.entry_link.token_value = token_value
        self.entry_link.token_id = token_id

    def get_exit_token(self):
        return self.exit_link.token_value, self.exit_link.token_id

    def propagate_entry_tokens(self, t):
        for link in [self.entry_link, *self.intm_links, self.exit_link]:
            link.push_token_to_entry(t)

    def propagate_exit_tokens(self, t, WLRs):
        for link in [self.entry_link, *self.intm_links, self.exit_link]:
            link.pull_token_from_exit(t, WLRs)


@dataclass
class ModelShell:
    #### NEED A METACLASS TO CREATE MODEL SHELLS!
    # 1) pass in a grammar
    # 2) pass in a set of hmm parameters A, avg, cov

    name: str
    num: int

    def __post_init__(self):
        self.model = None # SubSyntaxModel | HMM | None

    def set_grammar(self, grammar):
        self.grammar = grammar

    def set_HMMs(self, HMMs):
        self.HMMs = HMMs

    def set_sequence_data(self, sequence_data):
        self.sequence_data = sequence_data

    def spawn_SSM(self, ruleStr):
        possible_sequences = [elem.strip(' ') for elem in ruleStr.split('|')]

        ### first pass over sequence strings, to instantiate unique models
        instantiated_models = [] ### master list of instantiated models
        list_of_sequence_models = [] ### this list contains lists of references to instantiated models

        for sequence in possible_sequences:
            seqList = sequence.split(' ')

            sequence_models = []
            for model_name in seqList:
                prev_instances = [model.name == model_name for model in sequence_models]
                model_num = sum(prev_instances)

                ### instantiate new shell and copy top-level info
                new_model = ModelShell(model_name, model_num)
                new_model.set_grammar(self.grammar)
                new_model.set_HMMs(self.HMMs)
                new_model.set_sequence_data(self.sequence_data)

                if not new_model in instantiated_models: ### don't duplicate already instantiated models
                    instantiated_models.append(new_model)
                    sequence_models.append(new_model)
                else:
                    old_model = [mod for mod in instantiated_models if mod == new_model][0]
                    sequence_models.append(old_model)

            list_of_sequence_models.append(sequence_models)


        ### second pass over sequence models, to generate all binary links between model pairs
        binary_links = []
        for sequence_models in list_of_sequence_models:

            ### initalize the entry link -> no previous model
            prev_model = None

            ### go through models in sequence
            for model in sequence_models:

                binLink = BinaryLink(prev_model, model)
                if not binLink in binary_links:
                    binary_links.append(binLink)

                prev_model = model

            ### generate exit link
            binLink = BinaryLink(prev_model, None)
            if not binLink in binary_links:
                binary_links.append(binLink)

        ### check for dangerous two way links
        for binLink in binary_links:
            if binLink.reverse() in binary_links:
                if not binLink.is_entry() and not binLink.is_exit():
                    print(f'Warning: {ruleStr} implies two way binary link: {binLink}')

        ### third pass over preceeding models of binary links, to create half links
        prec_models = []
        for binLink in binary_links:
            precMod = binLink.prec
            if not precMod in prec_models:
                prec_models.append(precMod)

        # group binary links by preceding model
        half_links = []
        for precMod in prec_models:
            same_prec_links = [link for link in binary_links if link.prec == precMod]
            succList = [link.succ for link in same_prec_links] ## None succ -> exit link
            if precMod is None:
                entry_link = Link([], succList)
            else:
                halfLink = HalfLink(precMod, succList)
                half_links.append(halfLink)

        #### fourth pass over succeeding model lists of half links, to create full links
        succ_model_lists = []
        for halfLink in half_links:
            succMods = halfLink.succ
            if not succMods in succ_model_lists:
                succ_model_lists.append(succMods)

        # group half links by succeeding models
        exit_link_precs = []
        intm_links = []
        for succMods in succ_model_lists:

            same_succ_links = [link for link in half_links if link.succ == succMods]
            precMods = [link.prec for link in same_succ_links] ## None succ -> entry link

            if None in succMods: ### separate exit link from intermediate links
                exit_link_precs.extend(precMods) ### if None is in possible successors, all common precs are before exit.
                succMods.remove(None) ### then, to keep only intermediate material, remove None

            if not succMods == []:
                fullLink = Link(precMods, succMods)
                intm_links.append(fullLink)

        exit_link = Link(exit_link_precs, [])

        return SubSyntaxModel(entry_link, intm_links, exit_link)


    def inject_token(self, token_value, token_id):
        if token_value > np.log(0):

            if self.model is None: ### 'wakes up' the shell if a pos. prob. token is passed in
                if self.name.startswith('<') and self.name.endswith('>'): ### SSM
                    symbol = self.name.strip('<>')
                    expression = self.grammar[symbol]
                    self.model = self.spawn_SSM(expression)
                else: ### HMM
                    hmmDict = self.HMMs[self.name]
                    self.model = HMM(**hmmDict)
                    self.model.init_log_density(self.sequence_data)

            if self.model is not None: ### once we have a model, set entry tokens inside them
                self.model.set_entry_token(token_value, token_id)

    def propagate_entry_tokens(self, t):
        self.model.propagate_entry_tokens(t)

    def propagate_exit_tokens(self, t, WLRs):
        self.model.propagate_exit_tokens(t, WLRs)



#%% Function to make inference


def parse_grammar(text):
    """Parse BNF rules from text; blank lines and # comments are ignored."""
    rules = {}
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.count('::=') != 1:
            raise ValueError(f'Invalid grammar rule on line {number}')
        symbol, expression = (part.strip() for part in line.split('::=', 1))
        if not symbol or not expression or symbol in rules:
            raise ValueError(f'Empty or duplicate grammar rule on line {number}')
        rules[symbol] = expression
    if not rules:
        raise ValueError('Grammar must contain at least one rule')
    return rules


def read_grammar(path):
    """Read a grammar from an explicit filesystem path."""
    with open(path, encoding='utf-8') as handle:
        return parse_grammar(handle.read())


def work_back_WLR_links(modShell, WLRs):

    last_token_id = modShell.model.exit_link.token_id
    # WLRs = modShell.WLRs

    word_sequence = []

    while last_token_id not in [-1, 0]:
        t, num = last_token_id.split('_') ### a token id consists of time and an identifying number
        t, num = int(t), int(num)

        WLR = WLRs[t][num]
        exiting_model, period_exited = WLR['emitting_model'], WLR['t']

        if not (exiting_model.startswith('<') and exiting_model.endswith('>')):
            word_sequence.append((period_exited, exiting_model))

        last_token_id = WLR['token_id']

    return word_sequence

@np.errstate(divide='ignore')
def infer(sequence_data, HMMs, grammar, outer_grammar_keyword = '<all>'):

    ### initialize the outer model shell
    modShell = ModelShell(outer_grammar_keyword, 0)
    modShell.set_HMMs(HMMs)
    modShell.set_grammar(grammar)
    modShell.set_sequence_data(sequence_data)

    ### Initialize a WLR Dictionary
    WLRs = {key: {} for key in sequence_data.index}

    ### send a token of probability one to the outer grammar entry state
    modShell.inject_token(np.log(1), 0)

    for t in sequence_data.index:
        modShell.propagate_entry_tokens(t)
        modShell.propagate_exit_tokens(t, WLRs)

    ### Infer a sequence of events
    inferred_sequence = list(reversed(work_back_WLR_links(modShell, WLRs)))

    return inferred_sequence

def infer_mp(sequence_data_list, HMMs, grammar, outer_grammar_keyword = '<all>', num_workers = 24):
    from multiprocessing.pool import Pool
    from functools import partial

    partFunc = partial(infer, HMMs = HMMs, grammar = grammar, outer_grammar_keyword = outer_grammar_keyword)
    with Pool(num_workers) as p:
        outList = p.map(partFunc, sequence_data_list)

    return outList
