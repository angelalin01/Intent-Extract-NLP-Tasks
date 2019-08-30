import json
import nltk
import configparser
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions, ConceptsOptions, KeywordsOptions, EntitiesOptions, RelationsOptions, SemanticRolesOptions
from filestools import Tools

cf = configparser.ConfigParser()
cf.read("dbconf")
datafile = cf.get("file", "KB_content")
action_entities = cf.get("file", "aet")
read_output = Tools.read_csv_KB(datafile)
action_entity_output = Tools.read_csv(action_entities)
action_entity_output_string = []
for i in action_entity_output:
    s=i[0]
    action_entity_output_string.append(s)
read_output_relations_semantics = Tools.read_csv_KB_relations_semantics(datafile)
output = []


natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    iam_apikey='INJBZJ_fupvtsCswpI9UUOBZZmOYp6vHkHPEAZX_WU31',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
    )


def get_actions_related_to_entities():

    """ 
    FUNCTION: Perform semantic roles (POS tagging or identification of action, subject, object) in titles of articles to categorize action 
    entities with products/general entities they describe (and vice versa - match product/general entities with action entities that describe them)
        Represented by 2 dictionaries:
        - ACTS TO ENTS: which match actions to product/general entities 
                [action: [all event entities in sentence that action describes]]
        - ENTS TO ACTS: match product/general entities to actions that dsscribe them 
    
    STRUCTURE:
    PART 1:First calls IBM Watson semantic roles function to identify action, subject, object in each sentence, 
        but if IBM Watson is unable to perform function, go to PART 2

    PART 2: perform custom POS tagging engine 
        - first analyzes if there is (action + noun) bigrams sentence that does not contain ('or' + action) 
          because that indicates the verb only describes noun directly following it 

            e.g., DOES NOT CONTAIN 'OR': 'Create user accounts and set permissions' -- 'create' only describes 'user accounts' and 'set' only describes 'permissions'
            e.g., CONTAINS 'OR': 'Create, suspend, or cancel customer subscriptions' -- 'create', 'suspend', 'cancel' ALL describe 'customer subscriptions'
        
        - keeps track of the indices of the (action + noun) bigrams 
        - split the POS tag list by the indices and perform analysis within each sublist 
        - concatenates all the action and event entity words by "~" character
        - splits actions by "~" and matches event entities to each individual action 
            ***at the moment, only doing this for actions because usually actions are just one word, so each token in the actions list 
               concatenated by "~" can be assumed to be a different action whereas event entities are often more than one word, currently not 
               enough info on how to split event entities concatenated by "~"
    """

    actions_to_entities_dict = {}
    entities_to_actions_dict = {}

    #print sentences 
    for i in read_output:
        print(i[2])

    for i in read_output:
        url_link = i[0]
        title = i[2]
        content = i[3]
        content_sentences = i[3].split('.')
        title_tokens = title.split(' ')
        
        response = natural_language_understanding.analyze(
        text=title,
        features=Features(semantic_roles=SemanticRolesOptions())).get_result()
        json_dict = json.loads(json.dumps(response)) #turns json string output into dictionary
        
        #extract json values
        values_to_actions = [] # [event entities]
        values_to_entities = [] #[action entities]
        subject = ''
        action = ''
        object_string = ''
        entity = ''
        sentence = title

        #-----PART 1: PERFORM IBM WATSON SEMANTIC ROLES----

        if(json_dict['semantic_roles'] != []):
            print('ibm watson')
            sentence = json_dict['semantic_roles'][0]['sentence']
            if(json_dict['semantic_roles'][0]['subject']['text'] is not None):
                subject = json_dict['semantic_roles'][0]['subject']['text']
                print(subject)
            if(json_dict['semantic_roles'][0]['action']['text'] is not None):
                action = json_dict['semantic_roles'][0]['action']['text']
                print(action)
            if(json_dict['semantic_roles'][0]['object']['text'] is not None):
                print(object)
                object_string = json_dict['semantic_roles'][0]['object']['text']

            entity = entity + '~ ' + subject
            values_to_actions.append(entity)
            values_to_entities.append(action)
            #map each action/verb to the relavant entities; categorize entities to each action
            check_if_action_key_exists(action, actions_to_entities_dict, values_to_actions)
            check_if_entity_key_exists(entity, entities_to_actions_dict, values_to_entities)
        

        # ----- PART 2: PERFORM CUSTOM POS TAGGING IF WATSON FAILS ------
        else:
            tokens = nltk.word_tokenize(title)
            POS_tags = nltk.pos_tag(tokens)
            print(POS_tags)
            action_noun_template_count = 0
            bigram_dict = {}  
            bigrams = [] #keep track of bigrams to compare verb + noun templates
            contains_or_conjunction = False
            index_list = [] #list of indices by which POS tags will be split by 
            index_list.append(0)

            #populate bigrams based on criteria described above 
            for i,val in enumerate(POS_tags):
                if(i == len(tokens) - 1):
                    bigrams.append([val[1], ''])
                else:
                    bigrams.append([val[1], POS_tags[i+1][1]])
                    if(val[0].lower() == 'or' and (POS_tags[i+1][1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ') or POS_tags[i+1][1] in action_entity_output_string)):
                        contains_or_conjunction = True

            if contains_or_conjunction == False:
                for i, val in enumerate(bigrams): #i is index of POS tag in sentence 
                    first = val[0]
                    second = val[1]
                    if((first in ('VB', 'VBD', 'VBG', 'VBN', 'VBP') or first in action_entity_output_string) and second in ('NN', 'NNP', 'NNPS', 'NNS', 'CD')):
                        if(i != 0):
                            index_list.append(i)
            
            index_list.append(len(tokens) - 1)
            #go through POS_tags based on split indices to populate the dictionaries
            count = 0
            for i, index in enumerate(index_list):
                if(i < len(index_list) - 1):
                    if(len(index_list) > 2):
                        action = ''
                        entity = ''
                        values_to_actions = []
                        values_to_entities = []
                    for (word, tag) in POS_tags[index_list[i]:index_list[i+1]]:
                        if(word.lower() in action_entity_output_string): 
                            if(word == tokens[0]): #if first word of sentence is in action library, highly likely it is an action (nltk sometimes tags verbs as nouns)
                                action = action + '~ ' + word
                            else:
                                if(tag not in ('NNP', 'NNPS', 'NNS', 'NN', 'CD')): #need further clarification because POS tag isn't always most accurate
                                    action = action + '~ ' + word
                        elif(tag in ('NNP', 'NNPS', 'NNS', 'NN', 'CD') and word.lower() not in action_entity_output_string):
                            entity = entity + '~ ' + word #could be more than one product/general entity or entity is a phrase not a single word 
                    action = action.strip('~')
                    print(action)
                    entity = entity.strip('~')
                    if('~' in action): #handles scenarios in which multiple verbs are identified in sentence and we wamt to map each of them to the same noun entities identified
                        #e.g., "Create, suspend, or cancel customer subscriptions - Partner Center"-- we want ['Create' : [customer subscriptions, Partner Center]], ['suspend': [customer subscriptions, Partner Center]], ['cancel': [customer subscriptions, Partner Center]]
                        action_list = action.split('~')
                        for action_item in action_list:
                            values_to_entities.append(action_item)
                        values_to_actions.append(entity)
                        if(entity != ''):
                            check_if_entity_key_exists(entity, entities_to_actions_dict, values_to_entities)
                        for action_item in action_list:
                            if(action_item != ''): #see helper function
                                check_if_action_key_exists(action_item, actions_to_entities_dict, values_to_actions)

                    else:
                        values_to_entities.append(action)
                        values_to_actions.append(entity)
                        if(action != ''):
                            check_if_action_key_exists(action, actions_to_entities_dict, values_to_actions)
                        if(entity != ''):
                            
                            check_if_entity_key_exists(entity, entities_to_actions_dict, values_to_entities)
                        #print(values_to_actions)
    print('---- OUTPUT ----')
    print('ACTIONS MAPPED TO GEN/PROD ENTITIES')
    print(actions_to_entities_dict)          
    Tools.write_csv_data('./watson_output_actionsMap.csv',actions_to_entities_dict)
    print('GEN/PRDO ENTITIES MAPPED TO ACTIONS')
    print(entities_to_actions_dict)
    Tools.write_csv_data('./watson_output_entitiesMap.csv', entities_to_actions_dict)


#HELPER FUNCTIONS: updates dictionary with new keys and values 
def check_if_entity_key_exists(entity, entities_to_actions_dict, values_to_entities):
    if(entity not in entities_to_actions_dict.keys()):
        entities_to_actions_dict[entity] = values_to_entities
    else: #if action already exists in dictionary, add new values without overriding old values 
        for i in values_to_entities:
            entities_to_actions_dict[entity].append(i)


def check_if_action_key_exists(action, actions_to_entities_dict, values_to_actions):
    if(action not in actions_to_entities_dict.keys()):
        actions_to_entities_dict[action] = values_to_actions        
    else: #if action already exists in dictionary, add new values without overriding old values 
        for i in values_to_actions:
            actions_to_entities_dict[action].append(i)
     



""" ------------------------------------------------------------------------------------------------------------------------------------- """

#OTHER WATSON TEST FUNCTIONS
def test_watson():
    #print('hi')
    natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    iam_apikey='INJBZJ_fupvtsCswpI9UUOBZZmOYp6vHkHPEAZX_WU31',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
    )
    #print('xx')
    for i in read_output:
        #print('hello')
        url_link = i[0]
        print(url_link)
        #key categories
        #categories_response = natural_language_understanding.analyze(
        #url=url_link,
        #features=Features(categories=CategoriesOptions(limit=10))).get_result()
        #print(json.dumps(categories_response, indent=2))

        #key concepts
        concepts_response = natural_language_understanding.analyze(
        url=url_link,
        features=Features(concepts=ConceptsOptions(limit=10))).get_result()
        output.append(json.dumps(concepts_response, indent=2))
        print('CONCEPTS')
        print(json.dumps(concepts_response, indent=2))

        #top keywords
        keywords_response = natural_language_understanding.analyze(
        url=url_link,
        features=Features(keywords=KeywordsOptions(sentiment=True,emotion=False,limit=10))).get_result()
        output.append(json.dumps(concepts_response, indent=2))
        print('KEYWORDS')
        print(json.dumps(keywords_response, indent=2))

        #top entities
        entities_response = natural_language_understanding.analyze(
        url=url_link,
        features=Features(entities=EntitiesOptions(sentiment=False,limit=10))).get_result()
        output.append(json.dumps(concepts_response, indent=2))
        print('ENTITIES')
        print(json.dumps(entities_response, indent=2))

    Tools.write_csv_data('./watson_output.csv', output)

#semantic roles 
def test_example():
        
    natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    iam_apikey='INJBZJ_fupvtsCswpI9UUOBZZmOYp6vHkHPEAZX_WU31',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
    )

    
    response1 = natural_language_understanding.analyze(
    text='To use MPN support contracts, you need the Access ID and Contract ID for the support contract. ',
    features=Features(semantic_roles=SemanticRolesOptions())).get_result()
    #print(json.dumps(response1, indent=2))

    response2 = natural_language_understanding.analyze(
    text='Demonstrate your proven expertise in delivering quality solutions in one or more specialized areas of business. ',
    features=Features(semantic_roles=SemanticRolesOptions())).get_result()

    #print(json.dumps(response2, indent=2))

    response3 = natural_language_understanding.analyze(
    url='https://docs.microsoft.com/en-us/partner-center/',
    features=Features(semantic_roles=SemanticRolesOptions())).get_result()

    print(json.dumps(response3, indent=2))



def test_example_relations():
    natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    iam_apikey='INJBZJ_fupvtsCswpI9UUOBZZmOYp6vHkHPEAZX_WU31',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
    )

    response1 = natural_language_understanding.analyze(
    text='To use MPN support contracts, you need the Access ID and Contract ID for the support contract. ',
    features=Features(relations=RelationsOptions())).get_result()
    json_dict = json.loads(json.dumps(response1))
    print(json_dict['usage']['text_units'])
    print(json_dict['relations'])
    print('RELATIONS')
    #print(json.dumps(response1, indent=2))


    response2 = natural_language_understanding.analyze(
    text='Demonstrate your proven expertise in delivering quality solutions in one or more specialized areas of business. ',
    features=Features(relations=RelationsOptions())).get_result()
    print(json.dumps(response2, indent=2))




def test_watson_relations_semantics():
    #print('hi')
    natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    iam_apikey='INJBZJ_fupvtsCswpI9UUOBZZmOYp6vHkHPEAZX_WU31',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
    )
    string = 'Add licenses or services to an existing subscription'

  
    #print('xx')
    for i in read_output_relations_semantics:
        #print('hello')
        url_link = i[0]
        print(url_link)


        #top relations; this is generic just from watson's current library but we can customize later
        relations_response = natural_language_understanding.analyze(
        url=url_link,
        features=Features(relations=RelationsOptions())).get_result()
        output.append(json.dumps(relations_response, indent=2))
        print('RELATIONS')
        print(json.dumps(relations_response, indent=2))

        semantics_response = natural_language_understanding.analyze(
        url=url_link,
        features=Features(semantic_roles=SemanticRolesOptions())).get_result()
        output.append(json.dumps(semantics_response, indent=2))
        print('SEMANTICS')
        print(json.dumps(semantics_response, indent=2))   

    Tools.write_csv_data('./watson_output_relations_semantics.csv', output)  

if __name__ == '__main__':
    get_actions_related_to_entities()
