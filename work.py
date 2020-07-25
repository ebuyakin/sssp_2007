# scratchpad

from utils import *
from model_01 import Model


def get_model_params(model):
    model_params = {
                    # 'message': model.message,
                    'total_threshold': model.distance_threshold_multiplier,
                    'incremental_threshold': model.increment_threshold,
                    'semantic_noise': model.lexicalization_noise_level,
                    'syntactic_noise': model.collocations_noise_level,
                    'tree_depth': model.sub_tree_depth_probs
                    }
    return model_params


def set_model_parameters(model, params):
    model.distance_threshold_multiplier = params[0]
    model.increment_threshold = params[1]
    model.lexicalization_noise_level = params[2]
    model.collocations_noise_level = params[3]
    model.sub_tree_depth_probs = params[4]


def test_model():

    m = Model()

    pp(m.__dict__)  # print model attributes
    pp(dir(m))  # print model methods and attributes

    # basic operating routine for randomly generated message:
    m.create_language_from_ML1()  # generate language from ML1
    m.generate_random_message()  # generate random message

    pp(get_model_params(m))  # check out the current model parameters
    m.message  # the current message in its raw form.
    m.draw_message()  # draw message as a tree (the same current message visualized as a tree)
    m.reset_message()  # reset message for the next attempt of lexicalization
    m.lexicalize_message()  # lexicalize message
    len(m.sentence_set)  # length of the resulting sentence set
    [m.draw_tree(s) for s in m.sentence_set]  # draw the resulting sentence sent

    set_model_parameters(m, params=[0.7, 0.4, 0.5, 0.5, [0, 0, 0, 0, 0, 0, 1]])  # modify the parameters







