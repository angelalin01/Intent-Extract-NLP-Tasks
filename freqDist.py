import nltk
from math import log 
from filestools import Tools
import configparser
import textdistance
import re
from nltk.stem import WordNetLemmatizer
import sys 
import argparse
from nltk import FreqDist
from nltk import ngrams

cf = configparser.ConfigParser()
cf.read("dbconf")
datafile = cf.get("file", "datasource")
prodfile = cf.get("file", "prodsource")
actionEntityFile = cf.get("file", "aet")
productEntityFile = cf.get("file", "pet")
otherEntityFile = cf.get("file", "ote")
stopwordfile = cf.get("file", "pesw")
amb_ind_file = cf.get("file", "ambInd")
spellErrorTest = cf.get("file", "testSpell")
otherEvent = cf.get("file", "otherEvent") #library for general noun entities (events)
chatProdEnts = []
chatPRodEnts_POS = []
chatProdEnts_unique_filtered = []
all_words = []

#weights
nnp_weigh = 1.8 #proper noun, singular
nnps_weigh = 1.7 #proper noun, plural
nn_weigh = 1.3 #singular noun
nns_weigh = 1.2 #plural noun
cd_weigh = 1.1 #cardinal digit
single_confidence = 0.83
multiple_confidence = 1.0

def clean_event_entity(st1):
    #st1 is each row of the data.csv chat log
    if st1 != ' ':
        stopwordlist = Tools.get_txt_data(stopwordfile) #get stopwords
        for i in stopwordlist:
            try:
                if i == "'":
                    r2 = re.findall('[' + i + ']', st1.strip(), re.I)
                else:
                    r2 = re.findall('\\b' + i + '\\b', st1.strip(), re.I)
                if len(r2) > 0:
                    st1 = st1.replace(r2[0], '').strip() # replace stopword with whitespace?
            except:
                print(i)
    return st1

#get most frequent product entities based on new dataset that Levana sent (8/1)
def explore_freq_products2():
    prod_output2 = Tools.read_csv(spellErrorTest)
    general_ents = []
    for row in prod_output2:
        confidence=row[5]
        ent=row[4] #either product or general
        if(confidence=='High'):
            chatProdEnts.append(ent)
    freqOutput_prods = nltk.FreqDist(chatProdEnts)
    #print(freqOutput_prods)
    all_prod_words = freqOutput_prods.most_common(271) #based on how many samples there are
    greater_than_two = [sample[0] for sample in all_prod_words if sample[1] > 2]
    greater_than_one = [sample[0] for sample in all_prod_words if sample[1] > 1]
    print(len(greater_than_two))#169 words are have instances 3 and above 
    print(len(greater_than_one))#169 words are have instances 3 and above 
    #Tools.write_csv_data('./mostFrequentProds.csv', freqOutput_prods.most_common(169))

#get most frequent general entities based on new dataset Levana sent (8/1)
def explore_freq_gen():
    prod_output2 = Tools.read_csv(spellErrorTest)
    general_ents = []
    for row in prod_output2:
        confidence=row[5]
        ent=row[4] #either product or general
        if(confidence=='Middle'):
            general_ents.append(ent)
    freqOutput_gen = nltk.FreqDist(general_ents)
    gen_most_common = freqOutput_gen.most_common(1000)
    #all the general entities that occur more than twice
    greater_than_two = [sample[0] for sample in gen_most_common if sample[1] > 2]
    print(len(greater_than_two))
    Tools.write_csv_data('./mostFrequentGen.csv', freqOutput_gen.most_common(608))

#get most frequent words based on old chat transcript/dataset
def explore_freq_products():
    prod_output = Tools.read_csv(prodfile)
    for i in prod_output:
        s = i[0]
        sss = clean_event_entity(s)
        tokens = nltk.word_tokenize(sss)
        pos_tags = nltk.pos_tag(tokens)
        s1=''
        score=0
        for j in pos_tags:
            #we only care about certain tags for product entity identification
            if(j[1] == 'NNP') or (j[0].isupper()):
               # NNP_count = NNP_count+1
                s1 += j[0] + ' '
                score += nnp_weigh
            elif(j[1] == 'NNPS'): 
                #NNPS_count = NNPS_count+1
                s1 += j[0] + ' '
                score += nnps_weigh
            elif(j[1] == 'NN'):
                #NN_count = NN_count+1
                s1 += j[0] + ' '
                score += nn_weigh
            elif(j[1] == 'NNS'):
                #NNS_count = NNS_count+1
                s1 += j[0] + ' '
                score += nns_weigh
            elif(j[1] == 'CD'):
                #CD_count = CD_count+1
                s1 += j[0] + ' '
                score += cd_weigh
            else:
                if s1 != '' and score > 0:
                    s1_length = len(s1.strip().split(' ')) + 1
                    scored = log(score)/log(s1_length)
                    if(s1_length == 2 and scored > single_confidence) or (s1_length > 2 and scored > multiple_confidence):
                        s1 = s1.strip()
                        chatProdEnts.append(s1)

                s1 = ''
                score = 0
    freqOutput = nltk.FreqDist(chatProdEnts)
    #Tools.write_csv_data('./mostFrequent.csv', freqOutput.most_common(28))
    print(freqOutput.most_common(346))
    print(freqOutput)
    mostFreq = cf.get("file", "mostfreqProd")
    most_freq_prods = Tools.read_csv(mostFreq)
    most_freq_prods_words = []
    
    for i in most_freq_prods:
        most_freq_prods_words.append(i[0].lower())

    for k in chatProdEnts:
        if(k.lower() not in most_freq_prods_words):
            chatProdEnts_unique_filtered.append(k)

    print(most_freq_prods_words)
    print(chatProdEnts_unique_filtered)

    for k in chatProdEnts_unique_filtered:
        print(k)
        #print(k[0][0])

def explore_freq_acts2():
    act_output2 = Tools.read_csv(spellErrorTest)
    act_ents = []
    for row in act_output2:
        ent=row[3] #either product or general
        act_ents.append(ent)
    freqOutput_act = nltk.FreqDist(act_ents)
    print(freqOutput_act)
    #greater_than_two = [sample[0] for sample in all_words if sample[1] > 2]
    #print(len(greater_than_two))
    Tools.write_csv_data('./mostFrequentActs.csv', freqOutput_act.most_common(280))

#get most frequent action entities based on old dataset/chat transcript 
def explore_freq_acts():
    print('hello')
    read_output = Tools.read_csv(datafile)
    actionEntity_output = Tools.read_csv(actionEntityFile)
    chatActionEnts = []
    chatActionEnts_lemm = []

    lemmatizer = WordNetLemmatizer()

    for i in read_output:
        s = i[0]
        tokens = nltk.word_tokenize(s)
        pos_tags = nltk.pos_tag(tokens)
        for j in pos_tags:
            #we only care if the tag is a verb/action word 
            if(j[1] in ('VB', 'VBD', 'VBN', 'VBG', 'VBP', 'VBZ')):
                #lemm_word = lemmatizer.lemmatize(j[0], pos="v")
                orig_word = j[0]
                chatActionEnts.append(orig_word)
                #chatActionEnts_lemm.append(lemm_word)
    #analyze frequency distribution of both original and lemmatized words
    #Tools.write_csv_data('./chatActionEnts.csv', chatActionEnts)
    #Tools.write_csv_data('./chatActionEnts_lemm.csv', chatActionEnts_lemm)
    #chat_act_ents = cf.get("file", "chatActEnts")
    #chat_act_ents_lemm = cf.get("file", "chatActEntsLemm")
    #output = Tools.read_csv(chat_act_ents)
    #output_lemm = Tools.read_csv(chat_act_ents_lemm)
    result = FreqDist(chatActionEnts)
    #result_lemm = FreqDist(chatActionEnts_lemm)
    print(len(chatActionEnts))
    #print(result.most_common(250))
    Tools.write_csv_data('./mostFrequentActs.csv', result.most_common(2500))
    #Tools.write_csv_data('./mostFrequentActs_lemm.csv', result_lemm.most_common(250))
    all_words = result.most_common(875735)
    greater_than_two = [sample[0] for sample in all_words if sample[1] > 2]
    print(len(greater_than_two))
    #print(result)
    #print(result_lemm)

    #get all the act ents that have count greater than 1 
def get_greater_than_one_count():
    greater_than_three = [sample[0] for sample in all_words if sample[1] > 3]
    print(len(greater_than_three))
    #Tools.write_csv_data('./mostFreqActs_Multiple_count', greater_than_two)

if __name__ == '__main__':
    explore_freq_products2()
