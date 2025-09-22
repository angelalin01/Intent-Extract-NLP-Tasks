import json
import nltk
import configparser
import time
import random
import functools
import logging
import os
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, CategoriesOptions, ConceptsOptions, KeywordsOptions, EntitiesOptions, RelationsOptions, SemanticRolesOptions
from ibm_watson import ApiException
from filestools import Tools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


class WatsonRateLimiter:
    def __init__(self, requests_per_second=1):
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
        
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


rate_limiter = WatsonRateLimiter()


def watson_api_retry(max_attempts=3, base_delay=1, max_delay=60):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    rate_limiter.wait_if_needed()
                    return func(*args, **kwargs)
                except ApiException as e:
                    if e.code >= 400 and e.code < 500:
                        logger.error(f"Watson API client error (4xx): {e.code} - {e.message}")
                        raise
                    elif e.code >= 500 or e.code == 429:
                        if attempt == max_attempts - 1:
                            logger.error(f"Watson API server error after {max_attempts} attempts: {e.code} - {e.message}")
                            raise
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                        logger.warning(f"Watson API error (attempt {attempt + 1}/{max_attempts}): {e.code} - {e.message}. Retrying in {delay:.2f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"Watson API unexpected error: {e.code} - {e.message}")
                        raise
                except Exception as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"Watson API network/connection error after {max_attempts} attempts: {str(e)}")
                        raise
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(f"Watson API network error (attempt {attempt + 1}/{max_attempts}): {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


watson_api_key = os.getenv('WATSON_API_KEY', 'INJBZJ_fupvtsCswpI9UUOBZZmOYp6vHkHPEAZX_WU31')
if watson_api_key == 'INJBZJ_fupvtsCswpI9UUOBZZmOYp6vHkHPEAZX_WU31':
    logger.warning("Using hardcoded Watson API key. Consider setting WATSON_API_KEY environment variable for security.")

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    iam_apikey=watson_api_key,
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
        
        try:
            response = watson_analyze_semantic_roles(title)
            json_dict = json.loads(json.dumps(response))
        except Exception as e:
            logger.error(f"Watson API failed for title '{title}': {str(e)}. Falling back to custom POS tagging.")
            json_dict = {'semantic_roles': []}#turns json string output into dictionary
        
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
            logger.info('Using IBM Watson semantic roles analysis')
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
            logger.info('Watson API returned no semantic roles, falling back to custom POS tagging')
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


@watson_api_retry()
def watson_analyze_semantic_roles(text):
    return natural_language_understanding.analyze(
        text=text,
        features=Features(semantic_roles=SemanticRolesOptions())
    ).get_result()


@watson_api_retry()
def watson_analyze_concepts(url):
    return natural_language_understanding.analyze(
        url=url,
        features=Features(concepts=ConceptsOptions(limit=10))
    ).get_result()


@watson_api_retry()
def watson_analyze_keywords(url):
    return natural_language_understanding.analyze(
        url=url,
        features=Features(keywords=KeywordsOptions(sentiment=True, emotion=False, limit=10))
    ).get_result()


@watson_api_retry()
def watson_analyze_entities(url):
    return natural_language_understanding.analyze(
        url=url,
        features=Features(entities=EntitiesOptions(sentiment=False, limit=10))
    ).get_result()


@watson_api_retry()
def watson_analyze_relations(url):
    return natural_language_understanding.analyze(
        url=url,
        features=Features(relations=RelationsOptions())
    ).get_result()


@watson_api_retry()
def watson_analyze_relations_text(text):
    return natural_language_understanding.analyze(
        text=text,
        features=Features(relations=RelationsOptions())
    ).get_result()
     



""" ------------------------------------------------------------------------------------------------------------------------------------- """

#OTHER WATSON TEST FUNCTIONS
def test_watson():
    logger.info("Starting Watson API test functions")
    for i in read_output:
        url_link = i[0]
        print(url_link)
        
        try:
            #key concepts
            concepts_response = watson_analyze_concepts(url_link)
            output.append(json.dumps(concepts_response, indent=2))
            print('CONCEPTS')
            print(json.dumps(concepts_response, indent=2))
        except Exception as e:
            logger.error(f"Failed to analyze concepts for {url_link}: {str(e)}")

        try:
            #top keywords
            keywords_response = watson_analyze_keywords(url_link)
            output.append(json.dumps(keywords_response, indent=2))
            print('KEYWORDS')
            print(json.dumps(keywords_response, indent=2))
        except Exception as e:
            logger.error(f"Failed to analyze keywords for {url_link}: {str(e)}")

        try:
            #top entities
            entities_response = watson_analyze_entities(url_link)
            output.append(json.dumps(entities_response, indent=2))
            print('ENTITIES')
            print(json.dumps(entities_response, indent=2))
        except Exception as e:
            logger.error(f"Failed to analyze entities for {url_link}: {str(e)}")

    Tools.write_csv_data('./watson_output.csv', output)

#semantic roles 
def test_example():
    logger.info("Starting Watson semantic roles test examples")
    
    try:
        response1 = watson_analyze_semantic_roles('To use MPN support contracts, you need the Access ID and Contract ID for the support contract. ')
        logger.info("Successfully analyzed semantic roles for example 1")
    except Exception as e:
        logger.error(f"Failed to analyze semantic roles for example 1: {str(e)}")

    try:
        response2 = watson_analyze_semantic_roles('Demonstrate your proven expertise in delivering quality solutions in one or more specialized areas of business. ')
        logger.info("Successfully analyzed semantic roles for example 2")
    except Exception as e:
        logger.error(f"Failed to analyze semantic roles for example 2: {str(e)}")

    try:
        response3 = watson_analyze_semantic_roles('https://docs.microsoft.com/en-us/partner-center/')
        print(json.dumps(response3, indent=2))
        logger.info("Successfully analyzed semantic roles for example 3")
    except Exception as e:
        logger.error(f"Failed to analyze semantic roles for example 3: {str(e)}")
        print("Failed to analyze URL with Watson API")



def test_example_relations():
    logger.info("Starting Watson relations test examples")

    try:
        response1 = watson_analyze_relations_text('To use MPN support contracts, you need the Access ID and Contract ID for the support contract. ')
        json_dict = json.loads(json.dumps(response1))
        print(json_dict['usage']['text_units'])
        print(json_dict['relations'])
        print('RELATIONS')
        logger.info("Successfully analyzed relations for example 1")
    except Exception as e:
        logger.error(f"Failed to analyze relations for example 1: {str(e)}")

    try:
        response2 = watson_analyze_relations_text('Demonstrate your proven expertise in delivering quality solutions in one or more specialized areas of business. ')
        print(json.dumps(response2, indent=2))
        logger.info("Successfully analyzed relations for example 2")
    except Exception as e:
        logger.error(f"Failed to analyze relations for example 2: {str(e)}")
        print("Failed to analyze relations with Watson API")




def test_watson_relations_semantics():
    logger.info("Starting Watson relations and semantics test")
    string = 'Add licenses or services to an existing subscription'

    for i in read_output_relations_semantics:
        url_link = i[0]
        print(url_link)

        try:
            #top relations; this is generic just from watson's current library but we can customize later
            relations_response = watson_analyze_relations(url_link)
            output.append(json.dumps(relations_response, indent=2))
            print('RELATIONS')
            print(json.dumps(relations_response, indent=2))
        except Exception as e:
            logger.error(f"Failed to analyze relations for {url_link}: {str(e)}")

        try:
            semantics_response = watson_analyze_semantic_roles(url_link)
            output.append(json.dumps(semantics_response, indent=2))
            print('SEMANTICS')
            print(json.dumps(semantics_response, indent=2))
        except Exception as e:
            logger.error(f"Failed to analyze semantic roles for {url_link}: {str(e)}")

    Tools.write_csv_data('./watson_output_relations_semantics.csv', output)  

if __name__ == '__main__':
    get_actions_related_to_entities()
