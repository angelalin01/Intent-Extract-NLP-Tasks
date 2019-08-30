
import nltk
import textdistance
from math import log 
from Common.filestools import Tools
import configparser
import re
from nltk.stem import WordNetLemmatizer
import sys 
import argparse
from nltk import FreqDist
from nltk import ngrams
from Extract.SpellError.distanceedit import DistanceEdit
from Extract.SpellError.suffix import SuffixTree

"""
Initialize global variables start
"""
cf = configparser.ConfigParser()
cf.read("./Common/configfile")
datafile = cf.get("file", "datasource")
actionEntityFile = cf.get("file", "aet")
productEntityFile = cf.get("file", "pet")
freqWordFile = cf.get("file", "mostfreqAct")
spellErrorTest = cf.get("file", "testSpell")
otherEvent = cf.get("file", "otherEvent")
other_event = Tools.read_csv_chats(otherEvent)
read_output = Tools.read_csv_chats(datafile)
spell_test = Tools.read_csv_chats(spellErrorTest)
read_freq_words = Tools.read_csv(freqWordFile)
read_freq_words_words = [(i[0]).lower() for i in read_freq_words]
actionEntity_output = Tools.read_csv(actionEntityFile)
#print(actionEntity_output)
chatActionEnts = [] #all the words with verb POS tags 
chatActionEnts_unique = [] #all words in chatActionEnts not in the action entity library
chatActionEnts_filtered = [] #all words with verb POS tags that are not in the top 50 most frequent words
freqCount = []
sameWords_freqCount = []
same_words = []
levshDist = 2
levsh_ratio_thresh = 1/3 #Levsh ratio = levenshtein distance/length of correctly spelled word
#same_word_count = 0
potential_spell_error = 0
most_frequent = []
#freq measures of the type of verb POS tags 
VB_count = 0
VBD_count = 0
VBN_count = 0
VBG_count = 0
VBP_count = 0
VBZ_count = 0
same_words_VB = 0
same_words_VBD = 0
same_words_VBN = 0
same_words_VBG = 0
same_words_VBP = 0
same_words_VBZ = 0
#add a lemmatizer to make comparisons easier
lemmatizer = WordNetLemmatizer()
"""
Initialize global variables end
"""

class DetectionAction():

    def get_act_ents_from_chats(self): 
        """
        get the action entities from chat transcript 
        """
        for i in read_output:
            s = i[0]
            tokens = nltk.word_tokenize(s)
            pos_tags = nltk.pos_tag(tokens)
            for j in pos_tags:
                #we only care if the tag is a verb/action word 
                if(j[1] in ('VB', 'VBD', 'VBN', 'VBG', 'VBP', 'VBZ') and len(j[0]) >= 3 and j[0] not in read_freq_words_words):
                    chatActionEnts_filtered.append([j[0], j[1]])
                    if(j[1] == 'VB'):
                        VB_count = VB_count+1
                    if(j[1] == 'VBD'):
                        VBD_count = VBD_count+1
                    if(j[1] == 'VBN'):
                        VBN_count = VBN_count+1
                    if(j[1] == 'VBG'):
                        VBG_count = VBG_count+1
                    if(j[1] == 'VBP'):
                        VBP_count = VBP_count+1
                    if(j[1] == 'VBZ'):
                        VBZ_count = VBZ_count+1
        for k in chatActionEnts_filtered:
            lemm_word = lemmatizer.lemmatize(k[0], pos="v")
            chatActionEnts.append([lemm_word, k[1]])
    
        print('ALL IDENTIFIED VERB POS TAGS IN CHAT TRANSCRIPT')
        #print(chatActionEnts)
    
    def find_spell_check_error_actions_and_general(self,entities, library):
        same_words = []
        entities_unique = []
        same_word_count = 0
        belowLevshDistDICT = {} #close enough likely to be spelling error, or same word as word in action entity library 
        belowLevshDistLIST = [] # keep list for all the values, dictionary is just there to count number of UNIQUE chatActionEnts that are below Levsh dist from some action entity 
        afterFirstLetterFilter = [] #subset of the belowLevshDistLIST, only includes matches where first letters are the same 
        longest_substr_filter = []
        aboveLevshDist = []
        disedt=DistanceEdit()
        sufftree=SuffixTree()
        #only relevant for analyzing action entities in chat transcripts
        for k in entities:
            for i in library:
                if(i[0] == k[0]):
                    same_words.append(i[0])
                    same_word_count = same_word_count + 1
                    if(k[1] == 'VB'):
                        same_words_VB = same_words_VB + 1
                    if(k[1] == 'VBD'):
                        same_words_VBD = same_words_VBD + 1
                    if(k[1] == 'VBN'):
                        same_words_VBN = same_words_VBN + 1
                    if(k[1] == 'VBG'):
                        same_words_VBG = same_words_VBG + 1
                    if(k[1] == 'VBP'):
                        same_words_VBP = same_words_VBP + 1
                    if(k[1] == 'VBZ'):
                        same_words_VBZ = same_words_VBZ + 1
    
        #print('ALREADY IN LIBRARY')    
        #print(same_words)
        for k in entities:
            if(k[0] not in same_words):
                entities_unique.append(k)
        print('ENTITIES UNIQUE')
        print(len(entities_unique))
        print(entities_unique)
        for k in entities_unique:
            for i in library:
                if(i[0] == k[0]):
                    print('ERROR: STILL CONTAINS MATCHING VALUES')
                    print(k[0])
                else:
                    if(i[0] in k[0]):
                        longest_substr_filter.append([k[0], i[0]])
                        belowLevshDistDICT[k[0]] = i[0]
                      #APPLY LEVENSHTEIN DISTANCE FILTER 
                    #do ing filtering here 
                    dist = disedt.iterative_levenshtein(i[0], k[0])
                    levsh_ratio = dist/(len(i[0]))
                    if(dist <= levshDist and dist > 0 and levsh_ratio <= levsh_ratio_thresh):
                        key = k[0]
                        belowLevshDistDICT[key] = i
                        belowLevshDistLIST.append([k[0], i[0]])
                        #APPLY FIRST LETTER FILTER 
                        if(i[0][0] == k[0][0] and i[0][1] == k[0][1]):
                            afterFirstLetterFilter.append([k[0], i[0]])
                            #APPLY LONGEST SUBSTRING FILTER
                        
                            if(len(k[0]) <=3):
                                sufftree.append_string(i[0])
                                sufftree.append_string(k[0])
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 2 and len(i[0]) == len(k[0])):
                                    longest_substr_filter.append([k[0], i[0]])
                            elif(len(k[0]) <6):
                                sufftree.append_string(i[0])
                                sufftree.append_string(k[0])
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 3):
                                    longest_substr_filter.append([k[0], i[0]])
                            else:
                                sufftree.append_string(i[0])
                                sufftree.append_string(k[0])
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 4):
                                    longest_substr_filter.append([k[0], i[0]])
        print('AFTER FIRST LETTER FILTER')
        print(afterFirstLetterFilter)
        print('AFTER LONGEST SUBSTRING FILTER')
        print(longest_substr_filter)
        for j in entities:
            if(j[0] not in belowLevshDistDICT.keys() and j[0] not in same_words):
                aboveLevshDist.append(j[0])
        print('ABOVE LEVSH DIST')
        print(aboveLevshDist)
    
    def test_spell_check(self):
        print('MOST FREQ ACT ENTS')
        print(read_freq_words_words)
        print(len(read_freq_words_words))
        for row in spell_test:
            action=row[3]
            if(action not in read_freq_words_words):
                chatActionEnts_filtered.append([action, 'V'])
        print('ACT ENTS FILTERED')
        print(len(chatActionEnts_filtered))
        #only care about evaluating the lemmatized verbs to eliminate unnecessary, unmeaningful comparisons od "ing"/"ed"
        for k in chatActionEnts_filtered:
            lemm_word=lemmatizer.lemmatize(k[0], pos="v")
            chatActionEnts.append([lemm_word, 'V'])
        print(chatActionEnts)
        self.find_spell_check_error_actions_and_general(chatActionEnts, actionEntity_output)