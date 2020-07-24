# Language generator
"""
the class contains set of functions that can be used to generate language-like letter-based concepts store,
dictionary and semantic/syntactic associations between the concepts and the words.

The model builds generates a word-like set of concepts that have a pronounceable sets of letters associated with them.
In other words each abstract concept is associated with the string of letters from the alphabet.
This association is not related to the implied meaning of the concept, the only purpose of the association is to
create some visually clear association with the word.
The words are also randomly created from the same alphabet. The words and concepts are considered associated if
they 'sound similar' ie they are composed from similar letters. One possible interpretation is that letters (both
in concepts and words) actually represent some hidden (below awareness) semantic features that can be composed in
abstract conceptual store into the concepts or/and can be composed in dictionary into words.
The association of letters with concepts and words allows to express semantic similarity between the word and the
concept through their phonological similarity (thus making these associations easily identifiable).

Technically the phonological similarity (=semantic association) is measured by the notion of distance between
letter-strings corresponding to words and concepts.

The similarity is measured as sum of similarities between the letters. The similarity between the letters is
measured by 3 phoneme features for consonants and by 1 feature for vowels:

consonant letters type:
1. plosive (p,b,t,d,k,g),
2. fricative (f,v,s,z,h),
3. nasal (n,m),
4. liquid (l,r),
5. glide (w,j)

consonant letters positions:
1. bilabial (p,b,m,w)
2. labio-dental (f,v)
3. dental ()
4. alveolar (t,d,s,z,n,l)
5. post-alveolar (r,j)
6. velar (k,g,h)

consonant letter voice:
1. voiced (b,d,g,v,z,n,m,l,r,w,j)
2. voiceless (p,t,k,f,s,h)

vowels
a e i u o
"""

import numpy as np
from matplotlib import pyplot as plt
from random import sample


class ModelLanguage:

    def __init__(self):

        self.consonant_type = ['plosive', 'fricative', 'nasal', 'liquid', 'glide']

        # the distance between consonant_types. '01' is the distance between elements 0 and 1 of the list:
        self.consonant_type_distance = {'01': 1,
                                        '02': 3,
                                        '03': 3,
                                        '04': 3,
                                        '12': 3,
                                        '13': 3,
                                        '14': 3,
                                        '23': 1,
                                        '24': 1,
                                        '34': 1}

        self.consonant_position = ['bilabial', 'labio_dental', 'alveolar', 'post_alveolar', 'velar']
        self.consonant_position_distance = {'01': 1,
                                            '02': 2,
                                            '03': 3,
                                            '04': 3,
                                            '12': 1,
                                            '13': 2,
                                            '14': 3,
                                            '23': 1,
                                            '24': 2,
                                            '34': 1}

        self.consonant_voice = ['voiced', 'voiceless']
        self.consonant_voice_distance = 1

        self.consonants = {'b': [0, 0, 0],
                           'p': [0, 0, 1],
                           'd': [0, 2, 0],
                           't': [0, 2, 1],
                           'g': [0, 4, 0],
                           'k': [0, 4, 1],
                           'v': [1, 1, 0],
                           'f': [1, 1, 1],
                           's': [1, 2, 1],
                           'z': [1, 2, 0],
                           'h': [1, 4, 1],
                           'n': [2, 2, 0],
                           'm': [2, 0, 0],
                           'l': [3, 2, 0],
                           'r': [3, 3, 0],
                           'w': [4, 0, 0],
                           'j': [4, 3, 0]}

        self.vowels = ['a', 'e', 'i', 'u', 'o']

        self.vowels_distance = {'01': 1,
                                '02': 2,
                                '03': 3,
                                '04': 2,
                                '12': 1,
                                '13': 3,
                                '14': 3,
                                '23': 2,
                                '24': 3,
                                '34': 1}

        self.phoneme_frequency = {'a': 8,
                                  'e': 12,
                                  'i': 7,
                                  'o': 7,
                                  'u': 3,
                                  'b': 1.5,
                                  'd': 4.3,
                                  'f': 2.3,
                                  'g': 2,
                                  'h': 5.9,
                                  'j': 0.1,
                                  'k': 0.6,
                                  'l': 4,
                                  'm': 2.6,
                                  'n': 6.9,
                                  'p': 1.8,
                                  'r': 6,
                                  's': 6.2,
                                  't': 9.1,
                                  'v': 1.1,
                                  'w': 2.1,
                                  'z': 0.1}


    def distance_consonants(self, p1, p2):

        c1_type_code, c1_position_code, c1_voice_code = self.consonants[p1]
        c2_type_code, c2_position_code, c2_voice_code = self.consonants[p2]

        if c1_type_code < c2_type_code:
            type_distance = self.consonant_type_distance[str(c1_type_code) + str(c2_type_code)]
        elif c2_type_code > c1_type_code:
            type_distance = self.consonant_type_distance[str(c2_type_code) + str(c1_type_code)]
        else:
            type_distance = 1

        if c1_position_code < c2_position_code:
            position_distance = self.consonant_position_distance[str(c1_position_code) + str(c2_position_code)]
        elif c2_position_code > c1_position_code:
            position_distance = self.consonant_position_distance[str(c2_position_code) + str(c1_position_code)]
        else:
            position_distance = 1

        if c1_voice_code == c2_voice_code:
            voice_distance = 1
        else:
            voice_distance = 2

        return type_distance * position_distance * voice_distance - 1


    def distance_vowels(self, v1, v2):

        c1_code = self.vowels.index(v1)
        c2_code = self.vowels.index(v2)

        if c1_code < c2_code:
            return self.vowels_distance[str(c1_code)+str(c2_code)] - 1
        elif c2_code < c1_code:
            return self.vowels_distance[str(c2_code)+str(c1_code)] - 1
        else:
            return 0


    def distance_syllables(self, s1, s2):
        """
        distance between the syllables. the syllable has a format consonant-vowel-consonant
        :param s1:
        :param s2:
        :return: sum of distances between the phonemes in the syllable
        """
        ds = self.distance_consonants(s1[0],s2[0]) + self.distance_vowels(s1[1],s2[1])
        ds += self.distance_consonants(s1[2],s2[2])

        return ds


    def distance_words(self, w1, w2, syllables_rank_multiplier=1.3):
        """
        the distance between the words is calculated as a function of distances between the syllables weighted
        by the position of the syllable in the word. for each syllable of the shorter word the distance
        for each syllable of the longer word (weighted by position of the syllable of the longer word)
        is calculated and the minimum among them is considered as the
        metrics of the expression of the shorter word syllable (ie the best syllable of the longer word is
        the best expression and the distance between them is taken.
        The sum of expressions of syllables is taken as the distance between the words.
        :param w1: first word as a string of letters from the alphabet
        :param w2: second word as a string of letters from the alphabet
        :param syllables_rank_multiplier: coefficient to scale the distance between vowels in different positions
        :return: the real number representing the distance between w1 and w2
        """
        # w1 = 'baran'
        # w2 = 'baron'

        if len(w1) <= len(w2):  # create 2 lists of syllables the words consist of
            ws_short = [w1[a*2:a*2+3] for a in range(int((len(w1)-1)/2))]
            ws_long = [w2[a*2:a*2+3] for a in range(int((len(w2)-1)/2))]
        else:
            ws_short = [w2[a*2:a*2+3] for a in range(int((len(w2)-1)/2))]
            ws_long = [w1[a*2:a*2+3] for a in range(int((len(w1)-1)/2))]

        # print(ws_long)
        # print(ws_short)

        d = 0  # initialize the distance
        d_log = {}  # the log of distance computation

        for i, s1 in enumerate(ws_short):
            ds_min = -1
            d_log[i] = []
            for j, s2 in enumerate(ws_long):
                # weighted distance between syllables s1 and s2 (i-th of the short and j-th of the lohg word)
                d_ = self.distance_syllables(s1, s2) + syllables_rank_multiplier**(abs(i-j))
                d_log[i].append(d_)
                if d_ < ds_min or ds_min == -1:
                    ds_min = d_  # find the minimum distance between the short and the long syllables
            d += ds_min  # add it to the distance between the words.
        return d, d_log


    def word_generator(self, word_len=5):
        """
        generates a sting of letters according to the letters frequency distribution (set as parameter) and
        specific (word-like) structure : CVCVCVCVC (ie the word always starts with consonant and alternates
        between consonants and vowels and ends with another consonant.
        :param word_len: the length of the word to be generated.
        :return: word as string of letters
        """
        f_v = list(self.phoneme_frequency.values())[:5]  # vowels frequencies
        tf_v = sum(f_v)
        probs_v = [f/tf_v for f in f_v]  # scaled vowels frequencies
        f_c = list(self.phoneme_frequency.values())[5:]  # consonants frequencies
        tf_c = sum(f_c)
        probs_c = [f/tf_c for f in f_c]  # normalized consonants frequencies

        syl_len = int(word_len/2)
        len_c = range(len(f_c))
        len_v = range(len(f_v))
        w = []
        for i in range(syl_len):  # for a number of syllables
            w.append(np.random.choice(len_c, p=probs_c) + 5)  # add consonant (starts from 5 position in probs)
            w.append(np.random.choice(len_v, p=probs_v))  # add vowel
        w.append(np.random.choice(len_c, p=probs_c) + 5)  # add final consonant

        pl = list(self.phoneme_frequency.keys())  # phoneme frequency keys are letters
        word = [pl[k] for k in w]  # word is a list of keys (phonemes)
        return ''.join(word)  # join the letters to create the string


    def generate_corpus(self, n=100, word_len_probs=[0, 0, 0, 1, 0, 3, 0, 2, 0, 1]):
        """
        generates the list of strings (can be used as words or concepts)
        :param n: number of concepts/words to be generated
        :param word_len_probs: probability distribution of the word length
        :return: corpus (list of words)
        """

        wlt = sum(word_len_probs)
        len_probs_adj = [p/wlt for p in word_len_probs]  # scaled
        len_range = range(len(len_probs_adj))

        corpus = []
        for i in range(n):
            wl = np.random.choice(len_range, p=len_probs_adj)
            corpus.append(self.word_generator(wl))

        return corpus


    def generate_derivative_dictionary(self, concepts_store, n_words,
                                       proximity_setting=[0, 0, 1, 0, 2]):
        """
        generates the dictionary with the words at the given distance from the concepts.
        the words are generated by replacement of a certain number of letters in the concept.
        e.g. for the concept LABEK one-letter replacement can create a word MABEK or LOBEK,
        two-letter replacement KIBEK, LABOT etc. the less replacements are made the smaller
        the distance between the words and the concepts.
        proximity_setting  defines how many words of how many replacement shall be included
        in the derivatvie dictionary. E.g.:
        proximity_setting[i] == j means each concept shall have j words with i replacement

        :param concepts_store: the set of concepts to be approximated. corpus. list of concepts (letter-sets)
        :param n_words: number of words in the dictionary that will be created
        :param proximity_setting: how many related words to create for each level of proximity
        :return: derivative dictionary = list of words
        """

        def replace_letters(concept, i):
            """
            convenience function that replaces i letters in the concept. randomly choosing which letters to
            replace and what letters to use instead. the replacementes are ensured to be different from
            original letters.
            :param concept:
            :param i:
            :return:
            """
            concept_ = list(str.lower(concept))  # convert string into the list
            replaced_positions = np.random.choice(range(len(concept_)),i)  # positions of letters to replace

            # letters probability distributions (for vowels and consonants separately)
            f_v = list(self.phoneme_frequency.values())[:5]  # vowels frequencies
            tf_v = sum(f_v)
            probs_v = [f / tf_v for f in f_v]  # scaled vowels frequencies
            f_c = list(self.phoneme_frequency.values())[5:]  # consonants frequencies
            tf_c = sum(f_c)
            probs_c = [f / tf_c for f in f_c]  # normalized consonants frequencies

            for rp in replaced_positions:
                if concept_[rp] in self.vowels:
                    choose_vowel = np.random.choice(range(len(self.vowels)), p=probs_v)
                    while choose_vowel == concept_[rp]:  # the vowel shall be different
                        choose_vowel = np.random.choice(range(len(self.vowels)), p=probs_v)
                    concept_[rp] = self.vowels[choose_vowel]
                else:
                    choose_consonant = np.random.choice(range(len(self.consonants)), p=probs_c)
                    while choose_consonant == concept_[rp]:
                        choose_consonant = np.random.choice(range(len(self.consonants)), p=probs_c)
                    consonants_keys = list(self.consonants.keys())
                    concept_[rp] = consonants_keys[choose_consonant]
            return ''.join(concept_)


        dictionary = []
        for concept in concepts_store:
            for i, reps in enumerate(proximity_setting):  # i is the number of replacement, j the number of words
                for j in range(reps):  # add j words with i replacements.
                    dictionary.append(replace_letters(concept, i))

        if len(dictionary) > n_words:  # sample the dictionary to ensure some variability in semantic distances
            dictionary = sample(dictionary, n_words)
        return dictionary


    def generate_language(self, concept_store_size=20, dictionary_size=30,
                          is_connected=False, proximity_setting=[0, 0, 1, 0, 2]):
        """
        method generates 3 objects: conceptual store, dictionary and semantic matrix
        :param concept_store_size: how many concepts to create
        :param dictionary_size: how many words to create
        :param is_connected: whether the derivative or random dictionary to be used
        :param proximity_setting: for derivative dictionary the proximity setting is supplied
        :return: self.concepts, self.dictionary, self.semantics, self.collocations
        """

        def softmax(arr):  # convenience function. sofrmax. used to normalize the semantic connections (distances)
            return np.exp(arr)/sum(np.exp(arr))


        self.concepts = self.generate_corpus(n=concept_store_size)

        if is_connected:  # use derivative dictionary
            self.dictionary = self.generate_derivative_dictionary(self.concepts, dictionary_size, proximity_setting)
        else:  # use random dictionary
            self.dictionary = self.generate_corpus(n=dictionary_size)

        self.semantics = np.zeros((dictionary_size, concept_store_size))

        for i in range(dictionary_size):  # this loop takes a long time. that data is stored, see below
            for j in range(concept_store_size):  # semantics is the distance between the concept and the word
                self.semantics[i, j] = self.distance_words(self.concepts[j], self.dictionary[i])[0]

        for i in range(concept_store_size):  # apply softmax function
            self.semantics[:,i] = softmax(self.semantics[:,i])

        self.concepts = [str.upper(x) for x in self.concepts]

        self.collocations = np.zeros((dictionary_size, dictionary_size))
        for i in range(dictionary_size):  # collocations are set randomly for each word with zipf distribution
            syntactic_diversity_ratio = np.random.uniform(1.2, 2)  # coefficient for zipf distribution
            zipf_sample = np.array([1/syntactic_diversity_ratio**i for i in range(dictionary_size)])  #
            zipf_sample = np.random.permutation(zipf_sample)
            self.collocations[:, i] = zipf_sample


"""
# TEST

self = ModelLanguage()
self.generate_language()
self.dictionary  # list of words
self.concepts  # list of words
self.semantics.shape  # array
self.collocations.shape  # array

"""

