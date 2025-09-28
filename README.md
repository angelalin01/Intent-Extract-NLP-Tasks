# Intent-Extract-NLP-Tasks
Various Python-based NLP algorithms used to perform intent extraction from query inputs to customer support chatbot: spell check, co-reference resolution, word sense disambiguation, semantic role analysis in each sentence

Actual chat transcripts not included as they are proprietary Microsoft data. This repo merely provides a framework to analyze a certain type of customer query input data and accurately extract intents through NLP algorithms. Work done during internship at Microsoft. 

Assumptions: 
- To efficiently extract intents, we've identified a few templates that occur repeatedly in input query data; the one most relevant is the "action entity" + "event entity" query template, where in order to properly articulate their customer support needs, customer queries will typically compose of an action entity (usually a verb) followed by an event entity (usually a noun). An example is "How do I renew my Microsoft Action Pack" where the action entity is "renew" and "Microsoft Action Pack" is the event entity. 

## Spell Check
See SpellCheck Output Writeup PDF for in-depth explanation. 
detectaction.py 
detectevent.py 
Helper functions include distanceedit.py and suffix.py

## Disambiguation
disambiguation.py (includes coreference resolution, word sense disambiguation)
See disambiguation explanation PDF for more in-depth explanation.

## Natural Language Analysis
freqDist.py and explore_watson3.py
Functions using IBM Watson Natural Language Understanding API and Python NLTK to perform various natural language analysis (frequency of words, semantic roles etc.). 
See actual python files for explanation of functions.

## Environment Setup
This project requires an IBM Watson Natural Language Understanding API key to function properly.

### Setting up the Watson API Key
1. Copy the example environment file: `cp .env.example .env`
2. Edit the `.env` file and replace `your_watson_api_key_here` with your actual IBM Watson API key
3. Alternatively, set the environment variable directly: `export WATSON_API_KEY=your_actual_api_key_here`

**Security Note**: Never commit your actual API key to version control. The API key should only be stored in environment variables or secure configuration files that are not tracked by git.   
