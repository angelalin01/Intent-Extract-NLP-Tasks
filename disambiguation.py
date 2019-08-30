import stanfordnlp
from Common.filestools import Tools
import configparser
import nltk
from math import log
import textdistance
import re
from nltk.stem import WordNetLemmatizer
import sys
import argparse

cf = configparser.ConfigParser()
cf.read("./Common/configfile")
nlp = stanfordnlp.Pipeline()
chatlist = Tools.read_csv(cf.get("read", "oet"))
amb_ind_file = cf.get("file", "ambInd")


class WordDisambiguation:
    @classmethod
    def word_senses_disambiguation_pro(self, row):
        """
        reference words from product entity
        Input:
            :param row: sentence and word need to reference [:type->list]
            :param st_governor: reference word [:type->str]
            :param tag_word: which word need to find reference word [:type->str]
        return:
            :param st_combine: return combine reference words [:type->str]
        """
        st_governor, st_combine = '', ''
        rs = row[0].split('?')
        for r in rs:
            r = r.lower()
            doc = nlp(r)
            st2 = doc.sentences[0].tokens
            num = 0
            tag_word = row[1].split()[-1]
            for i in st2:
                if i.text.lower() == tag_word.lower():
                    num = i.words[0].governor
                    break
            if (st2[num - 1].words[0].xpos.find('NN') > -1 or
                    i.words[0].xpos.find('CD') > -1) and row[1].lower().find(
                        st2[num - 1].text.lower()) == -1:
                st_governor = st2[num - 1].text
            if st_governor != '':
                st_combine = st_governor
        if st_combine == '':
            return ''
        return st_combine.strip(';')

    @classmethod
    def word_senses_disambiguation(self, row):
        """
        reference words from other entity
        Input:
            :param row: sentence and word need to reference [:type->list]
            :param st_governor: reference word [:type->str]
            :param tag_word: which word need to find reference word [:type->str]
        return:
            :param st_combine: return combine reference words [:type->str]
        """
        st_governor, st_combine = '', ''
        rs = row[0].split('?')
        for r in rs:
            r = r.lower()
            doc = nlp(r)
            st2 = doc.sentences[0].tokens
            num = 0
            tag_word = row[1]
            for i in st2:
                if i.text.lower() == tag_word.lower():
                    num = i.words[0].index
                    break
            for i in st2:
                if i.words[0].governor == int(num) and (
                        i.words[0].xpos.find('NN') > -1
                        or i.words[0].xpos.find('CD') > -1):
                    st_governor += i.text + ' '
            if st_governor != '':
                st_combine = st_governor + '' + tag_word
        if st_combine == '':
            return tag_word
        return st_combine.strip(';')


class WordAmbiguity:
    def __init__(self):
        self.ambiguity_indicators = Tools.read_csv_ambiguity(amb_ind_file)
        self.tokens_prev = []
        self.pos_prev = []
        self.tokens_next = []
        self.pos_next = []


    def find_ambiguous_verb_reference(self, sentence):
        """
        INPUT: Takes in the sentences where ambiguous word occurs, and sentence right before and after, which contain the potential reference
            param sentence:[type: -> string]

        RETURNS: list of potential entities that the ambiguous reference is describing either from the current or previous sentence
            param return_likely [type:-> [ambiguous verb: [potential reference, specificity, distance between reference and verb]]]
        
        IDEA:
        1. Analyze both the current portion of the sentence before the occurrence of the ambiguous word and the previous sentence, 
        2. Identify all NN, NNS, NNP, and NNPS and 
        3. Evaluate which of these noun POS tokens are most likely what the ambiguous word is referring to.

        PROCEDURE: 
        Tokenizes and finds POS tags of each word in current sentence. 
        For each token of a verb POS, whether the following token is: 
            1. punctuation mark (if case 1, call find_prod_ents_in_sent)
            2. an ambiguous pronoun like 'it' or 'that' etc. (if case 2, calls find_ambiguous_pronoun_reference)

        **there could be other types of words following the verb but above 2 are what we care about
         """
        token = ''
        tag = ''
        next_token = ''
        reference_likely = [] #what is returned 

        #s0 = previous sentence, s1 = current sentence containing ambiguous word, s2 = next sentence 
        s0, s1, s2 = self.change(sentence)

        #Tokenize and POS tag all sentences 
        if (s0 != ''):
            tokens_prev = nltk.word_tokenize(s0)
            self.pos_prev = nltk.pos_tag(self.tokens_prev)
        if (s2 != ''):
            self.tokens_next = nltk.word_tokenize(s2)
            self.pos_next = nltk.pos_tag(self.tokens_next)
        tokens_curr = nltk.word_tokenize(s1)
        pos_curr = nltk.pos_tag(tokens_curr)

        #analyze each token in the current sentence 
        for i, val in enumerate(pos_curr):
            token = val[0]
            tag = val[1]
            #find token right after ambiguous verb 
            if (i < (len(pos_curr) - 1)):
                next_token = pos_curr[i + 1][0]
            if (tag in ('VB', 'VBD', 'VBN', 'VBG', 'VBP', 'VBZ')): #find the verb in the sentence
                #determine whether ambiguous verb is followed by punctuation or another word
                #CASE 1: followed by punctuation
                if (next_token == '' or next_token == '?' or next_token == '.'
                        or next_token == '!'):
                    #analyze current and previous sentence by finding product entities (potential references)
                    reference_likely_curr = self.find_prod_ents_in_sent(s1, i)
                    if (reference_likely_curr[0] ==
                            'no prod/gen entities in prev sentence'):
                        print('NO REFERENCE FOLLOWING VERB IN CURRENT SENTENCE. ANALYZE PREVIOUS')
                        #verb index is in relative position to words in prev sentence
                        #e.g., 'My suscription will expire soon. How do I renew?' -- verb index of renew = 8
                        reference_likely_prev = self.find_prod_ents_in_sent(
                            s0, i + len(tokens_prev))
                        reference_likely.append(token)
                        reference_likely.append(reference_likely_prev)
                    else:
                        reference_likely.append(token)
                        reference_likely.append(reference_likely_curr)
                #CASE 2: followed by ambiguous pronoun 
                elif (next_token in self.ambiguity_indicators):
                    print('AMBIGUOUS REFERENCE FOLLOWING VERB')
                    reference_likely_curr = self.find_ambiguous_pronoun_reference(
                        s0, s1, s2)
                    reference_likely.append('{0} {1}'.format(token,next_token))
                    reference_likely.append(reference_likely_curr)
        print(reference_likely)
        return reference_likely

    def find_prod_ents_in_sent(self, s0, verb_index):
        """
        INPUT: takes in index of the ambiguous verb and the sentence to be analyzed 
            param s0: [type: -> String]
            param verb_index: [type -> int]

        RETURNS: likely product or general entities (right now, words with NNP, NNPS, NN, NNS tag) 
                that the ambiguous verb is referring to
            return type: list 
                        [potential reference, specificity, distance from reference to verb]
        """
        tokens_prev = nltk.word_tokenize(s0)
        pos_prev = nltk.pos_tag(tokens_prev)
        reference_likely = []
        for i, value in enumerate(pos_prev):
            print(value)
            if (pos_prev[i][1] == 'NNP' or pos_prev[i][1] == 'NNPS'):
                dist_from_word = verb_index - i
                reference_likely.append(
                    [tokens_prev[i], 'high', dist_from_word])
            elif (pos_prev[i][1] == 'NN' or pos_prev[i][1] == 'NNS'):
                dist_from_word = verb_index - i
                reference_likely.append(
                    [tokens_prev[i], 'medium', dist_from_word])
        if (len(reference_likely) == 0):
            reference_likely.append('no prod/gen entities in prev sentence')
        return reference_likely

    def find_ambiguous_pronoun_reference(self, s0, s1, s2):
        """
        INPUT: takes in sentence containing ambiguous pronoun, the sentence right before, and the sentence right after
            param s0: previous sentence [type -> string]
            param s1: current sentence with ambiguous pronoun [type -> string]
            param s2: following sentence [type -> string]

        RETURNS: likely references to ambiguous pronouns 
            return type: list 
            [potential reference, specificity, 'curr_sent' or 'prev_sent', distance from reference to ambiguous word]
                    
            - 'curr_sent' or 'prev_sent' : whether reference occurs in the previous or current sentence
            -  with potential reference, which sentence it occurs, specificity (low-generic entity, medium- generic entity 
               directly after, high- product entity), and proximity to "it"/"that"
                    
        IDEA: performs evaluation criteria to determine which of the NN, NNS, NNP or NNPS is the pronoun referring to
        """
        if (s0 != ''):  
            self.tokens_prev = nltk.word_tokenize(s0)
            self.pos_prev = nltk.pos_tag(self.tokens_prev)
        if (s2 != ''):
            tokens_next = nltk.word_tokenize(s2)
            self.pos_next = nltk.pos_tag(tokens_next)
        tokens_curr = nltk.word_tokenize(s1)
        pos_curr = nltk.pos_tag(tokens_curr)

        reference_likely = []
        reference_likely_return = []
        index_amb_word_bool = False
        amb_word_counter = 0
        #list of the indices of all ambiguous references
        index_amb_word = []
        amb_word = ''

        #CASE 1: REFERENCE IS IN SAME SENTENCE AS AMBIGUOUS PROOUN
        for i, val in enumerate(tokens_curr):
            #find mention of ambiguous pronoun 
            if (tokens_curr[i] in self.ambiguity_indicators):
                index_amb_word_bool = True
                index_amb_word.append(i) #could be multiple ambiguous pronouns
                amb_word_counter = amb_word_counter + 1
                amb_word = tokens_curr[i]
                #CASE 1A: REFERENCE IS WORD RIGHT AFTER AMBIGUOUS PRONOUN 
                #check if following word is a product entity
                if (pos_curr[i + 1][1] is not None):  
                    dist = 1
                    if (pos_curr[i + 1][1] == 'NNP'
                            or pos_curr[i + 1][1] == 'NNPS'):
                        reference_likely.append(
                            [tokens_curr[i + 1], 'high', 'curr_sent', dist])
                    elif (pos_curr[i + 1][1] == 'NN'
                          or pos_curr[i + 1][1] == 'NNS'):
                        reference_likely.append(
                            [tokens_curr[i + 1], 'medium', 'curr_sent', dist])
        if (index_amb_word_bool == True):
            #assumes the actual reference to "it" or "that" occurs before "it"/"that" occurs
            #need to consider each ambiguous reference; val is the index of each amb reference
            for l, val in enumerate(index_amb_word):
                print(l)
                #all words before each ambiguous reference
                for j in range(1, val):
                    if (pos_curr[j][1] == 'NNP' or pos_curr[j][1] == 'NNPS'):
                        dist_from_word = index_amb_word - j
                        reference_likely.append([
                            tokens_curr[j], 'high', 'curr_sent', dist_from_word
                        ])
                    elif (pos_curr[j][1] == 'NN' or pos_curr[j][1] == 'NNS'):
                        dist_from_word = index_amb_word - j
                        reference_likely.append([
                            tokens_curr[j], 'low', 'curr_sent', dist_from_word
                        ])

                #CASE 2: REFERNECE IS IN PREVIOUS SENTENCE 
                for k, value in enumerate(self.pos_prev):
                    #print(pos_prev[k][1])
                    dist_from_word = len(self.pos_prev) + val - k
                    if (value[1] == 'NNP'or value[1] == 'NNPS'):
                        if (value[1] not in self.ambiguity_indicators):
                            reference_likely.append([
                                self.tokens_prev[k], 'high', 'prev_sent',
                                dist_from_word
                            ])
                    elif (value[1] == 'NN' or value[1] == 'NNS'):
                        if (value[1] not in self.ambiguity_indicators):
                            reference_likely.append([
                                self.tokens_prev[k], 'low', 'prev_sent',
                                dist_from_word
                            ])
                if (len(reference_likely) == 0):
                    # needs more analysis; maybe call find_ambiguous_pronoun_reference() again with new input
                    print('need to analyze sentence before prev sentence')
                    
                #sort by distances
                reference_likely.sort(
                    key=lambda reference_likely: reference_likely[3])  
                reference_likely_return.append(['ambiguous word:', amb_word])
                reference_likely_return.append(reference_likely)
                reference_likely = []
        return reference_likely_return

    def change(self, sentence):
        """
        Adjust the input sentences to a certain extent.
        """
        characters = ['?_.', '.com_@com', '!_.']
        for i in characters:
            sentence = sentence.replace(i.split('_')[0], i.split('_')[1])
        arr = sentence.split('.')
        if len(arr) >= 3:
            curr_sentence, prev_sentence, next_sentence = arr[0], arr[1], arr[
                2]
        elif len(arr) == 2:
            curr_sentence, prev_sentence, next_sentence = arr[0], arr[1], ''
        elif len(arr) == 1:
            curr_sentence, prev_sentence, next_sentence = '', arr[0], ''
        return curr_sentence, prev_sentence, next_sentence
