# EXPERIMENT 1
"""
This is a template experiment designed to test the architecture of the model and demonstrate how the model
can be applied to study various relationships between sentence production variables.
Experiment 1 investigates relationship between the distance threshold (which is a metrics of accuracy of
the sentence). The distance (semantic distance) measures how well the sentence expresses the meaning of the
message. By manipulating the distance threshold the model requests more accurate lexicalization of the message.
The hypothesis that is tested in the experiment is that greater accuracy requires longer time to achieve (ie
lower fluency of speech) and may require more complex syntactic structure of the sentence.
Note, that accuracy here is understood as 'semantic accuracy', not 'grammatical accuracy'. The same principle,
however, can be applied for analysis of grammatical accuracy, if the language that is used by the model is
supplemented by grammar rules.
"""

# general import:
import numpy as np
import pandas as pd
from itertools import product
from time import time, localtime, strftime

import importlib  # pycharm specific module required to correctly import model.
import model_01  # import of the model class
importlib.reload(model_01)  # to ensure pycharm reloads the model after it's modified. not needed for other IDEs.
from model_01 import Model  # re-import the latest version of the model class.
import reglam_utilities
importlib.reload(reglam_utilities)  # to ensure pycharm reloads the model after modified. not needed for other IDEs.
from reglam_utilities import *


# environment definition:
def set_environment(message_structure=['x', 'x0', 'x1', 'x01', 'x02'], n_messages=10):
    """
    function that creates the model and import the language.
    may be used to set some of the parameters of the model provided they are not manipulated in the experiment.
    :return: model
    """

    m = Model()  # create new instance of the Model class.
    # m.create_basic_language()  # if model built-in random language is used
    m.is_derivative_dictionary = True  # set the parameter of the language import: use derivative dictionary
    m.num_concepts = 100
    m.num_words = 150
    m.proximity_setting = [0, 0, 1, 1, 2]  # set proximity settings for derivative dictionary
    m.create_language_from_ML1()  # create language from language_01 class

    m.message_structure = message_structure  # set the message structure
    corpus = []  # initialize the list for corpus of messages
    for i in range(n_messages):
        m.generate_structured_message()  # generate structured message (according to the structure set above)
        corpus.append(m.message)

    return m, corpus


# trial definition:
def trials(message, n_trials=1, verbose=True, **kwargs):
    """
    one trial is one instance of the message lexicalization. the function implements a series of trials
    i.e. lexicalizations of the same message with the same parameters. parameters are passed as kwargs
    :param message: message to be tested
    :param n_trials: number of lexicalizations to produce
    :param verbose: print out the state of lexicalization during the run time
    :param kwargs: custom settings of the model (if any)
    :return: log_trial: summary table of lexicalization trials (summary data)
    :return: log_encoding_set: list of detail records (timelines) of each individual lexicalization (full data)
    """
    global m  # model

    # set model parameters in accordance with the trial parameters.
    model_parameters = m.__dict__.keys()  # list of attributes of the model (includes parameters and other variables)
    for kw in kwargs:
        if kw in model_parameters:
            m.__dict__[kw] = kwargs[kw]  # set the class attribute equal to the argument of the trial
            if verbose:
                print(kw, ' : ', kwargs[kw])

    # helper function to compute the final distance:
    def evaluate_total_distance(log_encoding):
        lg = log_encoding[log_encoding.record_type == 'lexicalization is finished']
        lg_failure = lg[lg.verdict == 'no more head words. failure']
        if len(lg_failure) == 0:
            return lg.distance.sum()
        else:
            return 'xxx'

    m.message = message  # set the message for the model

    # initialize data collection facilities:
    fields = ['message',
              'concept_level_weights',
              'threshold_per_concept', 'total_threshold', 'increment_threshold', 'semantic_noise', 'syntactic_noise',
              'time', 'sentence_set', 'verdict', 'distance', 'length_of_sentence', 'depth_of_sentence',
              'n_sub_trees', 'n_head_words', 'n_roots', 'n_leafs',
              'verdict_success','verdict_one_word', 'verdict_failure']  # fields
    log_trial = pd.DataFrame(columns=fields)  # initialize data frame to store experiment data
    log_encoding_set = []  # list to store the lexicalization record for each instance of lexicalization.

    if verbose:  # print out the trial run time report if needed
        print('   ')
        print('<<< run trials >>>')

    for i in range(n_trials):  # perform lexicalization n_trials times.
        print('trial: ', i)
        m.reset_message()  # reset message into non-lexicalized state
        m.lexicalize_message()  # encode the message

        if verbose:  # print out trial run time report if needed
            print('   ')
            print('<<< run message lexicalization: >>>')
            m.draw_message()  # print out the message tree (for illustration)
            ss = m.print_sentence_set()  # print out the sentence set in pseudo-natural format
            print(ss)

        log_encoding_set.append(m.log_encoding)  # save the log of individual lexicalization

        # record preliminaries:
        n_events = m.log_encoding.groupby(['record_type']).time.count()  # number of various events
        try:
            n_roots = n_events.loc['new_expansion_root']  # number of expansion roots tried
        except KeyError:
            n_roots = 0
        try:
            n_leafs = n_events.loc['new_leaf']  # number of expansion leafs tries
        except KeyError:
            n_leafs = 0

        length_of_sentence = sum([len(s) for s in m.sentence_set])  # calculate the number of words in the sentences
        depth_of_sentence = max([len(n['code']) for s in m.sentence_set for n in s])  # the number of levels
        if len(m.sentence_set) > 1:
            depth_of_sentence += 1  # multiple sentences are counted as one level of recursion

        total_distance = evaluate_total_distance(m.log_encoding)
        if total_distance != 'xxx':
            total_distance = str('%.3f' % total_distance)
        total_threshold = m.distance_threshold_multiplier * len(m.message)

        condensed_message = [n['label'] for n in m.message]
        condensed_message = ' '.join(condensed_message)

        last_event_index = m.log_encoding.index.max()  # max index of the encoding log (last record, last event)
        verdict = m.log_encoding.verdict[last_event_index]

        if verdict == 'success':
            verdict_flag = [1,0,0]
        elif verdict == 'success. one word sentence accepted':
            verdict_flag = [0,1,0]
        else:
            verdict_flag = [0,0,1]

        rec = {'message': condensed_message,
               'concept_level_weights': str(m.concept_level_weights),
               'threshold_per_concept': str('%.2f' % m.distance_threshold_multiplier),
               'total_threshold': str('%.2f' % total_threshold),
               'increment_threshold': str('%.2f' % m.increment_threshold),
               'semantic_noise': str('%.3f' % m.lexicalization_noise_level),
               'syntactic_noise': str('%.3f' % m.collocations_noise_level),
               'time': str('%.3f' % m.log_encoding.time[last_event_index]),
               'sentence_set': m.print_sentence_set(),
               'verdict': verdict,
               'distance': total_distance,
               'length_of_sentence': length_of_sentence,
               'depth_of_sentence': depth_of_sentence,
               'n_sub_trees': n_events.loc['new_sub_tree'],
               'n_head_words': n_events.loc['new_head_word'],
               'n_roots': n_roots,
               'n_leafs': n_leafs,
               'verdict_success': verdict_flag[0],
               'verdict_one_word': verdict_flag[1],
               'verdict_failure': verdict_flag[2]}
        log_trial = log_trial.append(rec, ignore_index=True)

    return log_trial, log_encoding_set


def block_of_trials(corpus, n_trials=5, verbose=False, **kwargs):
    """
    block of trials for different messages
    :param corpus: set of messages to be lexicalized
    :param n_trials: number of trials per message
    :param verbose: print out the state of the lexicalization during the run time
    :param kwargs: settings of the model
    :return: log_block: dataframe with the summary data for each trial.
    """

    global m

    for i, message in enumerate(corpus):
        print('\nmessage:', i)
        log_trial, log_encoding_set = trials(message, n_trials=n_trials, verbose=verbose, **kwargs)
        log_trial['message_id'] = str('%.0f' % i)
        if i != 0:
            log_block = pd.concat([log_block,log_trial])
        else:
            log_block = log_trial

    correct_order = list(log_block.columns)[-1:] + list(log_block.columns)[:-1]
    log_block = log_block[correct_order]

    return log_block


def experiment(manipulations, corpus, n_trials=5, verbose=False, **kwargs):
    """
    :param manipulations: dictionary of manipulated variables and their values
    :param corpus: set of messages to be lexicalized
    :param n_trials: number of trials per message
    :param verbose: flag for printing out the run time data
    :param kwargs: settings of parameters of the model
    :return:
    """

    global m
    param_space = list(product(*list(manipulations.values())))  # cartesian product of factor values

    for i, point in enumerate(param_space):
        params = {k: p for k, p in zip(manipulations, point)}  # dictionary of arg values for a given point
        print('\nparameter space point: ', i)
        kwargs_updated = {**params, **kwargs}  # merge 2 dictionaries
        print('factors: ', kwargs_updated)
        block_report = block_of_trials(corpus, n_trials=n_trials, verbose=verbose, **kwargs_updated)
        cols_order = ['factors_id'] + [p for p in params] + list(block_report.columns)
        block_report['factors_id'] = str(i)  # add id of the parameter space point
        for p in params:
            try:
                block_report[p] = params[p]
            except ValueError:
                print('params: ', params)
                print('cols_order: ', cols_order)
                print('block_report_columns: ', block_report.columns)
                print('p :', p)
                break 
        block_report = block_report[cols_order]  # change the order of column (for convenience of reading data)
        if i == 0:
            experiment_report = block_report
        else:
            experiment_report = pd.concat([experiment_report,block_report])

    return experiment_report





def run_simulations():

    message_structure = ['x', 'x0', 'x1', 'x2', 'x3', 'x01', 'x02', 'x03', 'x04',
                         'x10', 'x11', 'x12', 'x20', 'x21', 'x100', 'x1000',
                         'x200', 'x201']

    m, corpus = set_environment(message_structure=message_structure,n_messages=25)  # create model instance and
    # language instance

    print_messages = range(0, 25)
    if True:
        print('\ncorpus sample:\n')
        for i in print_messages:
            m.draw_tree(corpus[i])


    # run trial and collect the simulation data:
    message_index = 1
    tr, ls = trials(corpus[message_index], n_trials=3, verbose=False,
                    distance_threshold_multiplier=0.8,
                    increment_threshold=0.3,
                    lexicalization_noise_level=0.6,
                    collocations_noise_level=1.5)


    # run block of trials and collect data:
    br = block_of_trials(corpus, n_trials=2, verbose=False,
                         distance_threshold_multiplier=0.9,
                         increment_threshold=0.2,
                         lexicalization_noise_level=0.5,
                         collocations_noise_level=0.5)


    # run experiment

    manipulations = {'distance_threshold_multiplier': [0.4, 0.6, 0.8, 1.0],  # points
                     'increment_threshold': [0.1, 0.5],  # 9 points
                     'lexicalization_noise_level': [1],  # 5 points
                     'collocations_noise_level': [1],
                     'concept_level_weights': [[1, 1, 1, 1, 1]]}  # 2 points
    """
    manipulations = {'distance_threshold_multiplier': [0.6, 0.8],  # 7 points
                     'increment_threshold': [0.5],  # 9 points
                     'lexicalization_noise_level': [1],  # 5 points
                     'collocations_noise_level': [1.5]}  # 5 points
    """

    start_time, t1 = strftime('%H:%M:%S', localtime()), time()
    er = experiment(manipulations, corpus, n_trials=20)
    er.to_csv('xer_03.csv')
    end_time, t2 = strftime('%H:%M:%S', localtime()), time()
    print('time: ', start_time, ' - ', end_time, ' duration: ', float('%.2f' % (t2 - t1)))

    # the end

