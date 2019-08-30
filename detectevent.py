
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
levshDist = 3
same_word_count = 0
potential_spell_error = 0
levsh_ratio_thresh = 1/3
most_freq_prods_words = []
most_freq_gens_words = []
same_words = []
"""
Initialize global variables end
"""

class DetectionEvent():

    def test_spell_error_prod(self):
        """
        """
        general_ents = []
        for row in spell_test:
            confidence=row[5]
            #either product or general
            ent=row[4] 
            if(confidence=='Middle'):
                general_ents.append([ent])
        self.find_spell_check_error_actions_and_general(general_ents, other_event)

    def find_spell_check_error_actions_and_general(self,entities, library):
        """
        Helper method to find the action entities and general entities; same criteria applied to general entities as action entities
        INPUT: takes in entities to be evaluated and library to compare entities against 
            param entities: [type -> list]
            param library: [type -> list]
        Filtering process:
         - Filters out library words in the entities list (entities that match exactly one of the library words)
         - Filters out most frequent entity occurrences 
         - Apply Levenshtein filter, first letter filter, longest substring filter 
        """
        entities_unique = []
        entities_unique_filtered = []
        sameWords = []
        belowLevshDistDICT = {} #close enough likely to be spelling error, or same word as word in action entity library 
        belowLevshDistLIST = [] # keep list for all the values, dictionary is just there to count number of UNIQUE chatActionEnts that are below Levsh dist from some action entity 
        afterFirstLetterFilter = [] #subset of the belowLevshDistLIST, only includes matches where first letters are the same 
        longest_substr_filter = []
        aboveLevshDist = []
        disedt=DistanceEdit()
        sufftree = SuffixTree()
        print('GEN ENTS')
        print(len(entities))
        for k in entities:
            for i in library:
                if(i[0] == k[0]):
                    sameWords.append(i[0])
        print('SAME WORDS:')
        print(len(sameWords))

        for k in entities:
            if(k[0] not in sameWords):
                entities_unique.append(k)
                if(k[0] not in most_freq_gens_words):
                    entities_unique_filtered.append(k)
        print('GEN ENTS UNIQUE')
        print(len(entities_unique))
        print('GEN ENTS UNIQUE FILTERED')
        print(len(entities_unique_filtered))
        for k in entities_unique_filtered:
            for i in library:
                if(i[0] == k[0]):
                    print('ERROR: STILL CONTAINS MATCHING VALUES')
                    print(k[0])
                else:
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
        print('GENERAL ENTS: AFTER FIRST LETTER FILTER')
        print(afterFirstLetterFilter)
        print('GENERAL ENTS: AFTER LONGEST SUBSTRING FILTER')
        print(longest_substr_filter)
        for j in entities_unique:
            if(j[0] not in belowLevshDistDICT.keys() and j[0] not in same_words):
                aboveLevshDist.append(j[0])
        print('GENERAL ENTS: ABOVE LEVSH DIST')
        print(aboveLevshDist)

    def find_spell_error_prod(self,chatProdEnts, prodEntity_output):
        """
        Goes through all identified product entities and compares against both the synonym and entity in the library words
        INPUT: Takes in product entities and library words for product entities, in format of [synonym, entity]

        Apply filtering process: 
            - Filter out existing library words (both synonyms and actual entities) from identified product entities
            - Filter out most frequent occurring product entities 
            - The remaining words are compared against the library words using same Levenshtein, first two letter, longest substr filter
            CASE 1: POTENTIAL ENTITY MATCH CASE
            CASE 2: POTENTIAL SYNONYM MATCH CASE
                In either case, match to the entity because synonym means the same thing 

        """
        same_words = []
        same_word_count = 0
        chatProdEnts_unique = []
        chatProdEnts_unique_filtered = []
        belowLevshDistDICT = {} #close enough likely to be spelling error, or same word as word in action entity library 
        belowLevshDistLIST = [] # keep list for all the values, dictionary is just there to count number of UNIQUE chatActionEnts that are below Levsh dist from some action entity 
        afterFirstLetterFilter = [] #subset of the belowLevshDistLIST, only includes matches where first letters are the same 
        longest_substr_filter = []
        aboveLevshDist = []
        disedt=DistanceEdit()

        #filter through existing library words in the identified product entities 
        for k in chatProdEnts:
            k_lower = k.lower()
            for i in prodEntity_output:
                ent = (i[0]).lower()
                syn = (i[1]).lower()
                if(k_lower==ent):
                    same_words.append(ent)
                    same_word_count = same_word_count + 1
                elif(k_lower==syn):
                    same_words.append(syn)
                    same_word_count = same_word_count + 1

        print('CHAT PROD ENTS')
        print(len(chatProdEnts))
        for k in chatProdEnts:
            #compare everything in lowercase
            k_lower = k.lower()
            if(k_lower not in same_words):
                chatProdEnts_unique.append(k)
                if(k_lower not in most_freq_prods_words):
                    chatProdEnts_unique_filtered.append(k)
        print('CHAT PROD ENTS UNIQUE')
        print(len(chatProdEnts_unique))
        print('CHAT PROD ENTS UNIQUE FILTERED')
        print(len(chatProdEnts_unique_filtered))
        print(chatProdEnts_unique_filtered)
        for k in chatProdEnts_unique_filtered:
            k_lower = k.lower()
            for i in prodEntity_output:
                original = i[0]
                s = original.split(',')
                ent = (s[0]).lower()
                syn = (s[0]).lower()
                if(k_lower==ent or k_lower==syn):
                    print('ERROR: STILL CONTAINS MATCHING VALUES')
                    print(k)
                else:
                    #APPLY LEVENSHTEIN DISTANCE FILTER 
                    #cut out all "Microsoft" instances
                    if('microsoft' in k_lower):
                        k_no_microsoft = k.replace('microsoft', '')
                        k_lower = k_no_microsoft
                    if('microsoft' in ent):
                        ent_no_microsoft = ent.replace('microsoft', '')
                        ent = ent_no_microsoft
                    if('microsoft' in syn):
                        syn_no_microsoft = syn.replace('microsoft', '')
                        syn = syn_no_microsoft
                    #need to consider whether the identified product entity is synonym or actual entity
                    #compare against both syn and entity but for synonyms, match to entity 
                    dist = disedt.iterative_levenshtein(ent, k_lower)
                    dist_syn = disedt.iterative_levenshtein(syn, k_lower)
                    levsh_ratio_ent = 0
                    levsh_ratio_syn = 0
                    if(len(ent) != 0):
                        levsh_ratio_ent = dist/(len(ent))
                    if(len(syn) != 0):
                        levsh_ratio_syn = dist/(len(syn))

                    #CONSIDER ENTITY MATCH CASE 
                    if(dist <= levshDist and dist > 0 and levsh_ratio_ent > 0 and levsh_ratio_ent < levsh_ratio_thresh):
                        belowLevshDistDICT[k_lower] = ent
                        belowLevshDistLIST.append([k, ent])
                        #APPLY TWO LETTER FILTER 
                        if(ent[0] == k_lower[0] and ent[1] == k_lower[1]):
                            afterFirstLetterFilter.append([k, ent])
                            #APPLY LONGEST SUBSTRING FILTER
                            if(len(k) <=3):
                                sufftree = SuffixTree()
                                sufftree.append_string(ent)
                                sufftree.append_string(k)
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 2 and len(ent) == len(k[0])):
                                    longest_substr_filter.append([k, ent])
                            elif(len(k) <6):
                                sufftree = SuffixTree()
                                sufftree.append_string(ent)
                                sufftree.append_string(k)
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 3):
                                    longest_substr_filter.append([k, ent])
                            else:
                                sufftree = SuffixTree()
                                sufftree.append_string(ent)
                                sufftree.append_string(k)
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 4):
                                    longest_substr_filter.append([k, ent])

                    #CONSIDER SYNONYM MATCH CASE
                    #if there is a match with synonym, it is equivalent to matching with the entity
                    if(dist_syn <= levshDist and levsh_ratio_syn > 0 and dist_syn > 0 and levsh_ratio_syn < levsh_ratio_thresh):
                        key = k
                        belowLevshDistDICT[key] = syn
                        belowLevshDistLIST.append([k, syn])
                        #APPLY FIRST LETTER FILTER (compare against synonym)
                        if(syn[0] == k_lower[0] and syn[1] == k_lower[1]):
                            afterFirstLetterFilter.append([k, syn])
                            #APPLY LONGEST SUBSTRING FILTER
                        
                            if(len(k) <=3):
                                sufftree = SuffixTree()
                                sufftree.append_string(syn)
                                sufftree.append_string(k[0])
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 2 and len(syn) == len(k[0])):
                                    longest_substr_filter.append([k, ent])
                            elif(len(k) <6):
                                sufftree = SuffixTree()
                                sufftree.append_string(syn)
                                sufftree.append_string(k[0])
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 3):
                                    longest_substr_filter.append([k, ent])
                            else:
                                sufftree = SuffixTree()
                                sufftree.append_string(syn)
                                sufftree.append_string(k[0])
                                lcs = sufftree.find_longest_common_substrings()
                                if(len(lcs[0]) >= 4):
                                    longest_substr_filter.append([k, ent])
        print('PRODUCTS: AFTER FIRST LETTER FILTER')
        print(afterFirstLetterFilter)
        print('PRODUCTS: AFTER LONGEST SUBSTRING FILTER')
        print(longest_substr_filter)
        for j in chatProdEnts_unique_filtered:
            if(j not in belowLevshDistDICT.keys() and j not in same_words):
                aboveLevshDist.append(j)
        print('PRODUCTS: ABOVE LEVSH DIST')
        print(aboveLevshDist)