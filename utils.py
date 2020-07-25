# initial import and settings (template)

from nltk.corpus import wordnet as wn
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from time import time
from random import sample
from anytree import Node, RenderTree
from pprint import pprint as pp  # pretty printing


# console printout settings:
np.set_printoptions(precision=3, linewidth=200, suppress=True)
pd.set_option('display.max_columns', None)  # manage the number of columns that are printed into console
pd.set_option('display.precision', 3)  # floating point output precision (number of significant digits)
pd.set_option('display.width', None)  # the setting for the total width of the dataframe as it's printed.
pd.set_option('display.max_rows', 100)  # number of rows that is printed without truncation

# set matplotlib params
mpl.rcParams['figure.dpi'] = 300


# pretty printing function
def ppp(x, n=5):  # print list x by n items per line
    for i in range(int(len(x)/n)):
        print(x[i*n:i*n+n])
    print(x[(i+1)*n:])
    print('total: ', len(x))


def rmchart(x, n=50):  # standard plot with running mean option
    y = np.convolve(x, np.ones((n,)) / n)[(n - 1):]
    y = y[:-n]  # cut the tail
    plt.plot(y)
    plt.show()


def m_plot(x,y):
    plt.plot(x,y)
    plt.show()


def printer(x_list):  # convenience function to display a long list in the console
    for c in x_list:
        print(c)


def pr_tree(x, depth=3):  # prints an arbitrary multi-tier nested structure that consists of dictionaries and lists

    # x = [1,{'k1': 4, 'k2': [3, 5, {'l1': [3, 4, 5], 'l2': 555}], 'k3':5}, 3]  # example of the structure

    def ld(x, lvl=0):  # sub-function to convert lists within the nested structure into dictionaries with keys 'ind X'

        if type(x) is list:
            x_ = {'L' + str(lvl) + '-I' + str(i): elt for i, elt in enumerate(x)}  # convert into dictionary
        elif type(x) is dict:
            x_ = {'L' + str(lvl) + '-' + str(k): x[k] for k in x}

        if type(x_) is dict:
            for k in x_:  # loop over elements of the structure
                if type(x_[k]) is dict or type(x_[k]) is list:
                    lvl += 1
                    x_[k] = ld(x_[k], lvl=lvl)  # recursively call the function itself

        return x_

    x = ld(x)
    tree = list()

    def d_tree(x, parent='*'):  # makes a tree out of recursive dictionary

        nonlocal tree  # tree is function-level variable

        if parent == '*':  # top node of the tree
            parent = Node('*')
            tree.append(parent)

        counter = 0
        for k in x:
            if type(x[k]) is not dict:
                nd = Node(str(k) + ': ' + str(x[k]), parent=parent)  # the terminal node
                tree.append(nd)
            if type(x[k]) is dict:
                try:
                    k_level = int(k[1:k.find('-')])
                except AttributeError:
                    print(k)
                if k_level >= depth:
                    d_length = len(x)
                    nd = Node(str(k) + ': n_items = ' + str(d_length), parent=parent)
                else:
                    nd = Node(str(k), parent=parent)  # non-terminal node
                    tree.append(nd)
                    counter += 1
                    d_tree(x[k], parent=nd)  # recursive call of the function for the element of the dictionary
        return counter

    d_tree(x)  # apply the tree function

    for pre, fill, node in RenderTree(tree[0]):  # print the tree into console
        print('%s%s' % (pre, node.name))

    return x
