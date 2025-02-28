# Check if requirements.py has been run
confirmation = int(input('\033[1;33mHave you first run \'requirements.py\'? Enter 1 if yes, else 0.\nEnter: \033[0m'))
if confirmation == 1: print(f'\033[1;32mWe can proceed with the run.\033[0m')
else: print('\033[1;31mPlease run \'requirements.py\' first then come back again.\033[0m'); exit()

print(f'\033[1;33mLoading all libraries and setting up logging...\033[0m')
# 1. Libraries required for various purposes
print('\033[0;33mLoading general purpose libraries...\033[0m')
import os
import string
import ast
import json
import shutil
from datetime import date, datetime
import time
import pytz
from typing import Union

# 2. Stanza
print('\033[0;33mLoading Stanza...\033[0m')
import stanza

# 3. Textacy
print('\033[0;33mLoading Textacy...\033[0m')
import textacy

# 4. NLTK
print('\033[0;33mLoading NLTK..\033[0m')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk import RegexpParser

# 5. CoreNLP
print('\033[0;33mLoading CoreNLP...\033[0m')
corenlp_dir = './corenlp'
os.environ["CORENLP_HOME"] = corenlp_dir

# 6. Spacy
print('\033[0;33mLoading SpaCy and en_core_web_sm...\033[0m')
import spacy
nlp = spacy.load('en_core_web_sm')

# 7. Sentence Transformers
print('\033[0;33mLoading Sentence Transformers and util...\033[0m')
from sentence_transformers import SentenceTransformer, util

# 8. Data Science Libraries needed for the program
# The Pandas Library is used to load dataset into a dataframe
print('\033[0;33mLoading Data Science Libraries...\033[0m')
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import torch

print('\033[0;33mSetting up Logging...\033[0m')
import logging
logging.basicConfig(filename='log_file.log',
                    level=logging.INFO,
                    filemode='a',
                    force=True, # Resets any previous configuration, the most important thing to set in order to get our objective
                    )
# a million thanks to https://stackoverflow.com/questions/54597462/problem-with-logging-module-in-google-colab

print(f'\033[1;32mAll libraries loaded and logging setup successfully\033[0m')

def load_required_files_and_display_data() -> Union[pd.DataFrame, list, pd.DataFrame, list, list, 
                                                    list, dict, int, pd.DataFrame, list]:
    """
    Load the files required by the program, and display their data.

    Notes:
        The following files are required:
            sci310_Top-Sents_Set-per-Question.csv
            sci310_QuestionType.csv
            310_mcq.json
            answers-annots_310.
            annotations-para_310.json

    Returns:
        A tuple containing:
            df, all_sent, qtype_df, qtypes, mcq, finalmcq, finalpara, adjustedIndex, qns, questions
    """
    print('\033[1;33mLoading required files...\033[0m')
    # Read the sci310_Top-Sents_Set-per-Question.csv file into paras_df DataFrame
    set_per_q_file = "sci310_Top-Sents_Set-per-Question.csv"
    print(f'\033[0;33mLoading {set_per_q_file}...\033[0m')
    paras_df = pd.read_csv(f"./data/{set_per_q_file}")
    # Make a copy of paras_df into the df DataFrame
    df = paras_df

    print(f'{df.head()}')

    print(f"{df['new_paratext'][0]}")

    # qno, Question, Ans1, Ans2, Ans3, Ans4, support and correct_answer columns
    sci_para_mcq_file = "Sci-Para-mcqframed-Data-310.xlsx"
    print(f'\033[0;33mLoading {sci_para_mcq_file}...')
    qns = pd.read_excel(f'./data/{sci_para_mcq_file}')

    print(f'{qns.head()}')
    logging.info(f'qns.head():\n{qns.head()}\n')

    print(qns.loc[[0]])
    logging.info(f'qns.loc[[0]]:\n{qns.loc[[0]]}\n')

    questions = []

    for index, row in qns.iterrows():
        questions.append(row['Question'])

    print(f'\033[0;32mThe questions obtained from {sci_para_mcq_file} file are:\n{questions}\033[0m')
    logging.info(f'questions:\n{questions}\n')

    final_para = []

    for p in paras_df['new_paratext']:
        final_para.append('. '.join(list(ast.literal_eval(p))).replace('\n',"").lower())

    print(f'Printing the list of suporting sentences for the first row of the {set_per_q_file} file:\n{final_para[0]}')

    all_sent = []
    for i in final_para:
        j = i.lower()
        sentences_eachpara = j.split(".")
        all_sent.append(sentences_eachpara)  #index 0 para 1, indeexx 1 para 2 related sentences...
    question_type_file = "sci310_QuestionType.csv"
    print(f'\033[0;33mLoading {question_type_file}...\033[0m')
    qtype_df = pd.read_csv(f"./data/{question_type_file}")
    # Get the Question Type value for each entry in the sci310_QuestionType.csv file
    qtype_df = qtype_df["Question Type"]
    # Convert CSV to list
    qtypes = qtype_df.tolist()
    print(f'Number of Question Types before operation: {len(qtypes)}')
    logging.info(f'Number of Question Types before operation: {len(qtypes)}')
    count_dict = {}
    count_dict_all_categories = {}
    ctr = 0
    # The Question Types (qtypes) are 'what', 'how', 'which', 'why', 'where',
    # 'who' and 'when'
    for q in qtypes:
        if q in ['where','who','when']:
            qtypes[ctr]='other'
            count_dict['other'] = count_dict.get('other', 0) + 1
            ctr+=1
            count_dict_all_categories.update({q:count_dict_all_categories.get(q, 0) + 1})
        else:
            count_dict.update({q:count_dict.get(q,0)+1})
            count_dict_all_categories.update({q:count_dict_all_categories.get(q, 0) + 1})

    print(f'Dictionary of count: {count_dict}')
    logging.info(f'Dictionary of count: {count_dict}')
    print(f'Number of Question Types after operation: {len(qtypes)}')
    logging.info(f'Number of Question Types after operation: {len(qtypes)}')

    print(f'\033[1;33mNumber of \'which\' questions: {count_dict_all_categories["which"]}')
    print(f'Number of \'what\' questions: {count_dict_all_categories["what"]}')
    print(f'Number of \'how\' questions: {count_dict_all_categories["how"]}')
    print(f'Number of \'why\' questions: {count_dict_all_categories["why"]}')
    print(f'Number of \'where\' questions: {count_dict_all_categories["where"]}')
    print(f'Number of \'who\' questions: {count_dict_all_categories["who"]}')
    print(f'Number of \'when\' questions: {count_dict_all_categories["when"]}')
    print(f'Number of \'other\' questions: {count_dict_all_categories["where"] + count_dict_all_categories["who"] + count_dict_all_categories["when"]}\033[0m')

    # Load 310_mcq.json
    print(f'\033[0;33mLoading 310_mcq.json...\033[0m')
    f = open('./data/310_mcq.json')
    mcq = json.load(f)

    # Load answers-annots_310.json
    print(f'\033[0;33mLoading answers-annots_310.json...\033[0m')
    f = open('./data/answers-annots_310.json')
    finalmcq = json.load(f)
    
    # Load annotations-para_310.json
    print(f'\033[0;33mLoading annotations-para_310.json...\033[0m')
    f = open('./data/annotations-para_310.json')
    finalpara = json.load(f)

    # Update the 'mcq' variable to hold just the first item from the original JSON.
    mcq = mcq[0]
    # Access the first element of 'mcq', extract its question number and decrement
    # the value by 1.
    # Purpose is to change indexing from 1, ..., n to 0, ..., n-1
    adjustedIndex = mcq[0]["qno"] - 1
    print(f'The Adjusted Index is: {int(adjustedIndex)}')

    print('\033[1;32mAll files loaded\033[0m')
    
    return df, all_sent, qtype_df, qtypes, mcq, finalmcq, finalpara, adjustedIndex, qns, questions

def noun_chunks(data: str) -> list:
    '''
    Extracts noun phrases from a sentence or paragraph using SpaCy's chunking.

    Note:
        Uses the NLP Pipeline provided by SpaCy

    Arguments:
        data (str): The input sentence/paragraph from which noun phrases are to
        be extracted

    Returns:
        result (list): The list of noun phrases extracted from the input
        sentence/paragraph.
    '''

    data=data.replace('\n',' ')

    spacy_sent = list(nlp(data).sents)
    sentences = []
    for s in spacy_sent:
        x = s.orth_.strip()
        if len(x)!=0:
            sentences.append(x)

    print("Sentences : ",sentences)
    logging.info(f"Running the noun_chunks function...\nSentences:\n{sentences}")

    result = []

    for s in sentences:
        np = nlp(s).noun_chunks
        np = [t.text for t in np]
        result+=np

    return result

def get_pos_tag(word: str) -> str:
    '''
    This function is used to retrieve the POS tag of a given word by
    searching through tokenized sentences.

    Note:
        Uses ann, a parsed text structure from the Stanford CoreNLP package.

    Arguments:
        word (str): The word for which the POS tag is to be retrieved

    Returns:
        wd["pos"] (str): The POS tag of the word
    '''
    for sent in ann["sentences"]:
        for wd in sent["tokens"]:
            if wd["originalText"] == word:
                return wd["pos"]

def check_nmod(x: dict, nmoddict: dict) -> dict:
    '''
    This function identifies and stores nominal modifier differences (nmod)
    in a dictionary, mapping the governor words to their dependent words. It is
    useful for understanding relationships in a sentence such as possessives or
    prepositional modifiers.

    Arguments:
        x (dict): A dictionary containing information about the characteristics
        of the input sentence (CONFIRMATION NEEDED)
        nmoddict (dict): A dictionary storing the list of dependent words for
        each governor word

    Returns:
        nmoddict (dict): The updated nmoddict dictionary to reflect the data
        from the latest input sentence
    '''
    if x["dep"].startswith('nmod:'):
        if x["governorGloss"] not in nmoddict.keys():
            nmoddict.update({x["governorGloss"]:[x["dependentGloss"]]})
        else:
            nmoddict[x["governorGloss"]].append(x["dependentGloss"])
    else:
        return False

    return nmoddict

def check_obj(x: dict) -> bool:
    '''
    This function checks if a given token 'x' is an object in the sentence
    and if its governor is a verb. This helps identify object-verb relationships
    in a sentence, which is useful for dependency parsing.

    Arguments:
        x (dict): A dictionary containing information about the word

    Returns:
        A boolean value: True if all checks pass
    '''
    if x["dep"] == 'obj' and get_pos_tag(x["governorGloss"]) != None and get_pos_tag(x["governorGloss"]).startswith('VB'):
        return True
    return False

def check_nsubj(x: dict) -> bool:
    '''
    The function checks if a given token 'x' is a nominal subject in the
    sentence, and optionally if its governor is a verb. This is useful for
    dependency parsing to identify subject-verb relationships in sentences.

    Arguments:
        x (dict): A dictionary of the characteristics of the word

    Returns:
        A boolean value: True if all checks pass
    '''
    if (x["dep"] == 'nsubj' and get_pos_tag(x["governorGloss"]) != None and get_pos_tag(x["governorGloss"]).startswith('VB')) or (x["dep"] == 'nsubj'):
        return True
    return False

def check_vb_prep_n(x: dict) -> Union[str, bool]:
    '''
    This function checks for an oblique nominal (a noun related to a verb
    through a preposition) and extracts the preposition involved. This is helpful
    for identifying verb-preposition-noun structures in sentences, such as "go to
    the store" or "look at the sky".

    Arguments:
        x (dict): A dictionary of the details of the word

    Returns:
        pp (str): The preposition if found
        False (bool): Otherwise
    '''
    if x["dep"].startswith('obl:'):
        pp = x["dep"][4:]
        if get_pos_tag(pp) == 'IN':
            return pp
    return False

def check_prep_case(x: dict) -> bool:
    '''
    This function is used to determine whether a token in a sentence is a
    case marker, often related to prepositions that introduce noun phrases. It is
    helpful for identifying structures like prepositional phrases in a sentence.

    Arguments:
        x (dict): A dictionary of the information of the word

    Returns:
        A boolean value: True if the checks pass
    '''
    if x["dep"].startswith('case'):
        return True
    return False

def check_vb(x: dict) -> Union[str, bool]:
    '''
    This function checks if either the dependent or governor word in a
    dependency relation is a verb. It is useful for identifying verbs within
    specific grammatical structures in a sentence.

    Arguments:
        x (dict): A dictionary containing information about a sentence

    Returns:
        x['dependentGloss'] if the dependent word is a verb
        x['governorGloss'] if the governor word is a verb
        False (bool) otherwise
    '''
    if get_pos_tag(x["dependentGloss"]) != None and get_pos_tag(x["dependentGloss"]).startswith('VB'):
        return x["dependentGloss"]
    if get_pos_tag(x["dependentGloss"]) != None and get_pos_tag(x["governorGloss"]).startswith('VB'):
        return x["governorGloss"]
    return False

def check_nn(x: dict) -> Union[str, bool]:
    '''
    This function checks if either the dependent or governor word in a
    dependency relation is a noun. It is useful for identifying noun-related
    structures in sentences.

    Arguments:
        x (dict): A dictionary of the information of the input

    Returns:
        x['dependentGloss'], in case the dependent word is a noun
        x['governorGloss'], in case the governor word is a noun
        False (bool) otherwise
    '''
    print(f'Dependent Word: {x["dependentGloss"]}, Governor Word: {x["governorGloss"]}')
    if get_pos_tag(x["dependentGloss"]) != None and get_pos_tag(x["dependentGloss"]).startswith('NN'):
        return x["dependentGloss"]
    if get_pos_tag(x["governorGloss"]) != None and get_pos_tag(x["governorGloss"]).startswith('NN'):
        return x["governorGloss"]
    return False

def check_prep(x: dict) -> Union[str, bool]:
    '''
    Checks if the dependent word or the governor word is a preposition or not.

    Arguments:
        x (dict): A dictionary of the information of the input

    Returns:
        x['dependentGloss'], in case the dependent word is a preposition
        x['governorGloss'], in case the governor word is a preposition
        False (bool) otherwise
    '''
    if get_pos_tag(x["dependentGloss"]) != None and get_pos_tag(x["dependentGloss"])=='IN':
        return x["dependentGloss"]
    if get_pos_tag(x["governorGloss"]) != None and get_pos_tag(x["governorGloss"])=='IN':
        return x["governorGloss"]
    return False

def check_advb(x: dict) -> Union[str, bool]:
    '''
    Checks if the dependent word or the governor word is an adverb or not.

    Arguments:
        x (dict): A dictionary of the information of the input

    Returns:
        x['dependentGloss'], in case the dependent word is an adverb
        x['governorGloss'], in case the governor word is an adverb
        False (bool) otherwise
    '''
    if get_pos_tag(x["dependentGloss"]) != None and get_pos_tag(x["dependentGloss"]).startswith('RB'):
        return x["dependentGloss"]
    if get_pos_tag(x["governorGloss"]) != None and get_pos_tag(x["governorGloss"]).startswith('RB'):
        return x["governorGloss"]
    return False

def check_amod(x: dict) -> bool:
    '''
    This function checks whether a given dependency relation 'x' is an
    adjectival modifier 'amod' and whether the dependent word in the relation
    is an adjective (JJ). It is useful for identifying adjectives that modify
    nouns in a sentence, which can be helpful in tasks like sentence parsing,
    sentiment analysis or extracting descriptive information.

    Arguments:
        x (dict): A dictionary of the information of the input

    Returns:
        A boolean value, True if all checks pass
    '''
    if x["dep"] == 'amod':
        if get_pos_tag(x["dependentGloss"]) != None and get_pos_tag(x["dependentGloss"]).startswith('JJ'):
            return True
    return False

def removepun(stringinp: str) -> str:
    '''
    This function is used to clean up the text by removing various
    character, in a process called sanitization, to ensure that
    minor variations don't influence result. Basically all the
    punctuation symbols alongwith some extra symbols are removed.

    Arguments:
        stringinp (str): The string to be cleaned

    Returns:
        stringinp (str): The cleaned string
    '''
    for character in string.punctuation:
        stringinp = stringinp.replace(character, '')
    stringinp = stringinp.replace('\n',"")
    stringinp = stringinp.replace('\n ',"")
    stringinp = stringinp.replace(' \n ',"")
    stringinp = stringinp.replace('\n ',"")
    stringinp = stringinp.replace('\\n',"")
    stringinp = stringinp.replace(' \\n',"")
    stringinp = stringinp.replace('\t',"")

    return stringinp.replace(" ", "")

def wnLemma_v(word: str) -> str:
    '''
    This function is used to obtain the base or dictionary form (lemma)
    of a verb.

    Note:
        Uses the WordNetLemmatizer function of the NLTK Library

    Arguments:
        word (str): The verb to be lemmatized

    Returns:
        lem (str): The lemmatized version of the verb
    '''
    lem = WordNetLemmatizer().lemmatize(word, 'v')
    return(lem)

def wnLemma_n(word: str) -> str:

    '''
    This function is used to obtain the base or dictionary form (lemma)
    of a noun.

    Note:
        Uses the WordNetLemmatizer function of the NLTK Library

    Arguments:
        word (str): The noun to be lemmatized

    Returns:
        lem (str): The lemmatized noun
    '''
    lem = WordNetLemmatizer().lemmatize(word, 'n')
    return(lem)

def cosincode(parano: int, quesno: int, ansno: int, 
              all_sent: list, mcq: list, adjustedIndex: int, 
              model: SentenceTransformer) -> float:
    '''
    For the question whose Question Number is the same as the provided
    Paragraph Number, the list of sentences corresponding to that question
    is retrieved and encoded. Further for that question, the Answers of the
    list of questions corresponding to that Paragraph Number is retrieved and
    embedded. The cosine similarity between the answer embeddings and sentence
    embedding is calculated. The highest score is returned.

    Notes:
        It is assumed that a suitable Sentence Transformer model is available

    Arguments:
        parano (int): The paragraph number, used to retrieve the list of
        paragraphs
        quesno (int): The question number
        ansno (int): The answer number, used to retrieve the current answer

    Returns:
        final_cs (float): The final Cosine Similarity score for the best answer
    '''
    logging.info('Logs for the cosincode() function...')
    for qnas in mcq:
        if qnas["qno"] == int(parano):
            print(f'parano: {parano}, adjustedIndex: {adjustedIndex}')
            print(f'len(all_sent) in cosincode:{len(all_sent)}')
            para_lst = all_sent[int(parano) - 1 - int(adjustedIndex)]
            print("para list == ", para_lst)
            logging.info(f'para list == \n{para_lst}\n')
            sentence_embeddings = model.encode(para_lst)
            # POINT OF CONTENTION
            # IN CASE RESULTS ARE OFF, CHECK THIS LINE
            # OLD: each_ques = mcq[int(para_no) - 1 - int(adjustedIndex)]
            each_ques = mcq[int(parano) - 1 - int(adjustedIndex)]
            ans_set = each_ques["Answers"]
            curr_ans = ans_set[ansno - 1]
            ans_embeddings = model.encode(curr_ans)
            csarray = cosine_similarity([ans_embeddings], sentence_embeddings)
            print("cs array ans-emb and sent emb ==", csarray)
            logging.info(f"cs array ans-emb and sent emb == \n{csarray}\n")
            final_cs = np.max(csarray)
            print("final_cs", final_cs)
            logging.info(f"final_cs:\n{final_cs}")
            return final_cs

def find_cos(str1: str, str2: str, model: SentenceTransformer) -> float:
    '''
    This function finds the cosine similarity between the top
    sentences and the answers. It returns the maximum score.

    Notes:
        Cosine Similarity API is provided by the Model

    Arguments:
        str1 (str): The list of top sentences
        str2 (str): The list of answers

    Returns:
        final_cs (float): The highest value of Cosine Similarity
        obtained for a sentence-answer pair.
    '''
    topsentence_embeddings = model.encode(str1)
    ans_embeddings = model.encode(str2)
    csarray = cosine_similarity([ans_embeddings],[topsentence_embeddings])
    final_cs = np.max(csarray)
    return final_cs

def find_ann(sent: str, ann: dict) -> dict:
    '''
    This function accepts a string to be analyzed and an annotation object that
    contains linguistic annotations. It combines two key sources of noun-related
    information: Noun chunks and syntactic dependencies.

    Notes:
        Noun chunking is done using Spacy's noun chunker
        The annotation object is from Stanford CoreNLP

    Arguments:
        sent (str): The sentence to be analyzed
        ann (dict): The Stanford CoreNLP Annotation Object

    Returns:
        A dictionary containing:
            noun (list): The noun list
            nnchunks (list): The list of noun chunks
    '''
    logging.info('Start of logs for the find_ann() function\n')
    nnc = noun_chunks(str(sent).lower())
    print("------------")
    logging.info('----------------')
    sent_list = ann["sentences"]
    nounlist = []

    for t2 in sent_list:
        openie_list = t2["enhancedPlusPlusDependencies"]
        li2 = []
        for x in openie_list:
            if x["dep"] == "ROOT" or x["dependentGloss"] == '.' or x["governorGloss"] == '.':
                continue
            if check_vb_prep_n(x):
                nounlist.append(wnLemma_n(x["dependentGloss"]))
            if check_nn(x):
                nounlist.append(wnLemma_n(check_nn(x)))
            nounlist = list(set(nounlist))
            nnc = list(set(nnc))
    logging.info('End of logs for the find_ann() function\n')
    return {'noun': nounlist, 'nnchunks': nnc}

def common_no(list1: list, list2: list) -> Union[int, list]:
    '''
    This function determines how many and which elements from a list is
    also present in another list.

    Arguments:
        list1 (list): A list of items
        list2 (list): Another list of items

    Returns:
        A tuple containing:
            c (int): Count of the number of common elements
            lst (list): List of the common elements
    '''
    c = 0
    lst = []
    for list2_element in list2:
        if list2_element in list1:
            c += 1
            lst.append(list2_element)
    return c, lst

def getlist(string: str) -> list:
    '''
    This function performs cleanup of the input string,
    produces a list out of it by seperating segments on the
    basis of the comma, and returns it

    Arguments:
        string (str): The string to be processed

    Returns:
        li (list): The list of sections of the string,
        seperated on the basis of the comma
    '''
    string = string.replace("{", '')
    string = string.replace("}", '')
    string = string.replace("[", '')
    string = string.replace("]", '')
    string = string.replace("\\n", "")
    string = string.replace("'", '')
    li = [i.strip() for i in str.split(",")]
    return li

def getFramedOpts(quesn: int, qns: pd.DataFrame) -> list:
    '''
    This function is used later in the code during the cosine similarity
    scoring and prediction processes. It provides the answer options for a
    given question, which are then compared against predicted answers (using
    SBERT embeddings and cosine similarity).

    Arguments:
        quesn (int): The index of the question under consideration

    Returns:
        framedoptions (list): The list of answers (Ans1, Ans2, Ans3 and Ans4)
        for the question
    '''
    framedoptions = []
    framedoptions.append(qns.iloc[quesn-1]['Ans1'].replace('\n', ''))
    framedoptions.append(qns.iloc[quesn-1]['Ans2'].replace('\n', ''))
    framedoptions.append(qns.iloc[quesn-1]['Ans3'].replace('\n', ''))
    framedoptions.append(qns.iloc[quesn-1]['Ans4'].replace('\n', ''))
    return framedoptions

def getCorpus(quesn: int, df: pd.DataFrame, embedder: SentenceTransformer) -> tuple:
    '''
    This function gets the list of supporting sentences for the given
    question and also encodes them. Finally it returns both.

    Notes:
        The encoder is provided by a Sentence Transformer model.

    Arguments:
        quesn (int): The index of the question under consideration

    Returns:
        A tuple containing:
            corpus: The list of supporting sentences for the question
            corpus_embeddings: The encoded form of the corpus
    '''
    corpus = list(ast.literal_eval(df['new_paratext'][quesn-1]))
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)
    return corpus_embeddings, corpus

def getScore(quesn: int, embedder: pd.DataFrame,
             qns: pd.DataFrame, df: pd.DataFrame) -> list:
    """
    This function retrieves the answer options and supporting sentences
    for a given question quesn. For each answer, the supporting sentence
    most similar to it is identified. A list of tuples containing the
    current answer in question, the most similar supporting sentence and
    the similarity score is returned.

    Notes:
        quesn is assumed to be 1-based (first answer at index 1)
        Uses Cosine Similarity to get the simlarity between answer and
        supporting sentence.
        Returns top matching sentence even if the similarity is low

    Arguments:
        quesn (int): The index of the question in consideration

    Returns:
        final_answer (list): A list of tuples. Each tuple consists of
        an answer, the corresponding most similar supporting sentence
        and the similarity score
    """
    framedOptions = getFramedOpts(quesn, qns)
    corpus_embeddings, corpus = getCorpus(quesn, df, embedder)

    final_answer = []

    answers = framedOptions
    top_k = min(1, len(corpus))

    for answer in answers:
        answer_embedding = embedder.encode(answer, convert_to_tensor=True)

        cos_scores = util.cos_sim(answer_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            a = (answer, corpus[idx], format(score))
            final_answer.append(a)

    return final_answer

def similarity_scoring_using_edge_matching(mcq: list, finalmcq: list, finalpara: dict, 
                                           adjustedIndex: int) -> dict:
    """
    Perform Correct and Wrong Match Analysis.

    Notes:
        Uses the CoreNLP Client, which needs 8GB of System RAM and uses Port 9000.
        English Language support only.

    Arguments:
        mcq (list): Data from 310_mcq.json is loaded into this variable. See documentation for more details.
        finalmcq (list): Data from answers-annots_310.json is loaded into this variable. See documentation for more details. 
        finalpara (dict): Data from annotations-para_310.json is loaded into this variable. See documentation for more details.
        adjustedIndex (int): Obtained from load_required_files_and_display_data()
        
    Requires:
        load_required_files_and_display_data() 
        common_no()
        cosincode()
        removepun()

    Returns:
        A dictionary, containing:
            "edge_match": Matched questions using edge matching
            "pred_dict": Dictionary of predictions
            "edge_score": A list. See docs for more details.
            "mcq_score": A list. See docs for more details.            
    """

    edge_match = []
    pred_dict = {}

    from stanza.server import CoreNLPClient, StartServer
    CUSTOM_PROPS = {"parse.model": "edu/stanford/nlp/models/srparser/englishSR.beam.ser.gz"}

    # A tuple to store the current paragraph and question number
    x = ()

    logging.info('Starting Logs for Similarity Scoring using Edge Matching')
    # open a connection to the CoreNLP server for processing text
    # annotations with specific properties and annotators.
    with CoreNLPClient(properties = CUSTOM_PROPS,
                    output_format = "json",
                    annotators = ['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'coref', 'depparse'],
                    endpoint = 'http://localhost:9006',
                    be_quiet = True) as client:

        '''
        #1
        '''
        edge_score = []

        '''
        #2
        '''
        for ques_dict in finalmcq:
            count = 1
            para_no = str(ques_dict["qno"])

            '''
            #3
            '''
            for questions in ques_dict["Answers"]:
                print('--------------------------------------------------')
                print(f"\033[1;32mQuestion No.:{count}, Paragraph No.: {para_no}\033[0m")
                logging.info('--------------------------------------------------')
                logging.info(f"\033[1;32mQuestion No.:{count}, Paragraph No.: {para_no}\033[0m")
                '''
                #4
                '''
                mcq_score = []

                '''
                #5
                }
                '''
                for ques in questions:
                    # Store the current paragraph and question number
                    x = (int(para_no), count)
                    print(f"\033[1;32mxxxxxxxxxxxxxxx {x}\033[0m")
                    logging.info(f"xxxxxxxxxxxxxxx {x}\n")
                    # Store the score for the correct answer
                    score = 0
                    # List to store the matching syntactic elements between the
                    # paragraph and the answer
                    correctmatch = []
                    # Retrieves the Answer Number (1, 2, 3 or 4) for the
                    # current question
                    ans_no = ques["ans"]
                    # Retrieves the full question-and-answer data from the
                    # 'mcq' dataset using 'para_no'
                    qna = mcq[int(para_no) - 1 - int(adjustedIndex)]

                    print(f"\033[1;32mMCQ option.: {ans_no}\033[0m")
                    logging.info(f"MCQ option.: {ans_no}\n")
                    print(f"\033[1;32mFramed answer: {qna['Answers'][ans_no-1]}\033[0m")
                    logging.info(f"Framed answer: {qna['Answers'][ans_no-1]}\n")

                    '''
                    #6
                    '''
                    score += common_no(finalpara[para_no]["nsubj"],
                                    ques["nsubj"])[0]
                    score += common_no(finalpara[para_no]["dobj"],
                                    ques["dobj"])[0]
                    score += common_no(finalpara[para_no]["vprep"],
                                    ques["vprep"])[0]
                    score += common_no(finalpara[para_no]["nprep"],
                                    ques["nprep"])[0]
                    score += common_no(finalpara[para_no]["verb"],
                                    ques["verb"])[0]
                    score += common_no(finalpara[para_no]["prep"],
                                    ques["prep"])[0]
                    score += common_no(finalpara[para_no]["adjv"],
                                    ques["adjv"])[0]
                    score += common_no(finalpara[para_no]["subj_obj"],
                                    ques["subj_obj"])[0]
                    score += common_no(finalpara[para_no]["noun"],
                                    ques["noun"])[0]

                    '''
                    #7
                    '''
                    for i in finalpara[para_no]["nnchunks"]:
                        j = i.split()
                        for k in ques["nnchunks"]:
                            m = k.split()
                    score += common_no(m, j)[0]

                    print(f"\033[1;32mThe Total Score for this Answer Option is {score}\033[0m")
                    mcq_score.append(score)

                edge_score.append(mcq_score)
                count += 1

    print(f'\033[0;32mThe edge scores are as follows:\n{edge_score}\033[0m')
    logging.info(f'Edge Score:\n{edge_score}\n')
    logging.info('End of Logs for Similarity Scoring using Edge Matching')

    return {
            "edge_match": edge_match,
            "pred_dict": pred_dict,
            "edge_score": edge_score,
            "mcq_score": mcq_score
            }

def getBertAndEdgeScore(bert_score:list, edge_score:list) -> float:
    '''
    This function accepts a list of tuples containing the bert score
    for each answer option - context sentence combination and a list of tuples
    containing the edge score for each answer option - context sentence
    combination. It computes the sum of these scores. If this sum is the largest
    observed, the corresponding bert score is returned.

    Arguments:
        bert_score (list): a list of tuples containing the bert score
        for each answer option - context sentence combination
        edge_Score (list): a list of tuples containing the edge score
        for each answer option - context sentence combination

    Returns:
        maxx[0] (float): The bert score for which the total score is maximum

    '''
    maxx = (-1, -1, -1)
    logging.info('Inside getBertAndEdgeScore()...')
    norm_edge_score = normalize(np.array([edge_score])).tolist()[0]
    for i in range(len(bert_score)):
        print('--------------------------------')
        logging.info('--------------------------------')
        print(f"\033[1;34mBERT Score: {float(bert_score[i][2])}")
        logging.info(f"BERT Score: {float(bert_score[i][2])}")
        print(f"\033[1;34mEdge Score: {float(norm_edge_score[i])}")
        logging.info(f"Edge Score: {float(norm_edge_score[i])}")
        new_score = float(bert_score[i][2]) + float(norm_edge_score[i])
        if new_score > float(maxx[2]):
            maxx = (bert_score[i][0], bert_score[i][1], new_score)
    logging.info('Exiting getBertAndEdgeScore()...')
    return maxx[0]

def get_score_from_sbert_and_edge_matching(output: list, edge_score: list, qns: pd.DataFrame,
                                           qtypes: list, pred_dict: dict):
    """
    Returns:
        "unmatched": unmatched,
        "bert_match": bert_match,
        "bert_qtype": bert_qtype,
        "wrong": wrong,
        "corr_qna": corr_qna,
        "incorr_qna": incorr_qna
    """
    unmatched = []
    bert_match = []
    bert_qtype = {}
    wrong = []
    corr_qna = []
    incorr_qna = []

    count = 1
    for i in range(len(output)):
        print(f"\033[1;36m--------------Question {i+1}--------------\033[0m")
        logging.info(f"--------------Question {i+1}--------------")
        pred_opt = getBertAndEdgeScore(output[i], edge_score[i])
        cleaned_predicted_answer = removepun(str(pred_opt).lower())
        cleaned_actual_answer = removepun(str(qns['correct_answer'][i]))
        print(f'\033[1;33mPredicted Answer: {str(pred_opt)}\033[0m')
        print(f'\033[1;33mActual Answer: {str(qns["correct_answer"][i])}\033[0m')
        if cleaned_predicted_answer == cleaned_actual_answer.lower():
            print('\033[1;32mCorrect Match\033[0m')
            corr_qna.append((int(i+1), qns["Question"][i], qns["correct_answer"][i]))
            print((int(i+1), qns["Question"][i], qns["correct_answer"][i]))
            logging.info((int(i+1), qns["Question"][i], qns["correct_answer"][i]))
            bert_match.append((i+1, 1))
            bert_qtype.update({qtypes[i]: bert_qtype.get(qtypes[i], 0) + 1})
            pred_dict.update({qtypes[i]: pred_dict.get(qtypes[i], 0) + 1})
        else:
            print('\033[1;31mIncorrect Match\033[0m')
            logging.info('Incorrect Match')
            incorr_qna.append((int(i+1), qns["Question"][i], str(pred_opt), qns["correct_answer"][i]))
            print((int(i+1), qns["Question"][i], str(pred_opt), qns["correct_answer"][i]))
            logging.info((int(i+1), qns["Question"][i], str(pred_opt), qns["correct_answer"][i]))
            wrong.append([int(i+1), count, qns["Question"][i], qns["correct_answer"][i], str(pred_opt)])
            unmatched.append((int(i+1), 1))

    return {
        "unmatched": unmatched,
        "bert_match": bert_match,
        "bert_qtype": bert_qtype,
        "wrong": wrong,
        "corr_qna": corr_qna,
        "incorr_qna": incorr_qna
    }

def analyze_output(corr_qna: list, incorr_qna: list, 
                   bert_match: list, bert_qtype: dict,
                   unmatched: list, wrong: list) -> pd.DataFrame:
    """
    This function is used to analyse the output obtained by running the 
    correct_and_wrong_match_analysis() function. It prints the table of 
    Correct MCQs which includes Question No., Question and Correct Ans.
    columns. It also prints the table of Incorrect MCQs which includes
    Question No., Question Predicted Ans. and Correct Ans.

    Requires:
        correct_and_wrong_match_analysis()

    Arguments:
        matchedques (list): The list of matched questions. Each entry is a tuple (para_no, count).
        edge_match (list): The list of questions matched using edge matching. Similar structure as matchedques.
        edge_qtype (dict): A dictionary wherein the key is question type and value is number of matched questions
        with that question type
        corr_qna (list): A list of tuples storing the Index, Question and Answer of correctly matched question
        wrong_paras (list): A list of paragraph numbers for the incorrectly matched questions.
        incorr_qna (list): A list of tuples storing the Index, Question and Predicted Answer of incorrectly matched question
        wrong_match (list): A list of scores for the incorrectly matched questions.
        wrong_score (list): A list of incurr_corr (the exact elements that match between the paragraph
        and the answer) the incorrectly matched questions.

    Returns:
        unmatched_df (pd.DataFrame): The DataFrame of unmatched questions
    """
    print('\033[1;33mAnalyzing output...\033[0m')

    print("\033[1;32m--- Correct MCQs ---\nQuestion No. || Question ==> Correct Ans.\n\033[0m")
    logging.info("--- Correct MCQs ---\nQuestion No. || Question ==> Correct Ans.\n")
    for i in corr_qna:
        print("\033[1;32m", i[0], "||", i[1], "==>", i[2], "\033[0m")
        logging.info(f"\n{i[0]} \|\| {i[1]} ==> {i[2]}\n")

    print(f"The incorrectly matched questions are:\n{unmatched}\n")
    logging.info(f"The incorrectly matched questions are:\n{unmatched}\n")

    print("\033[1;31m--- Incorrect MCQs ---\nQuestion No. || Question ==> Predicted Ans. / Correct Ans.\n\033[0m")
    logging.info("--- Incorrect MCQs ---\nQuestion No. || Question ==> Predicted Ans. / Correct Ans.\n")
    for i in incorr_qna:
        print("\033[1;31m", i[0], "||", i[1], "==>", i[2], " / ", i[3], "\033[0m")
        logging.info(f"\n{i[0]} || {i[1]} ==> {i[2]} / {i[3]}\n")

    print(f"\033[1;32mNumber of questions matched using edge plus bert: {len(bert_match)}\033[0m")
    logging.info(f"Number of questions matched using edge plus bert: {len(bert_match)}")

    print(f"\033[1;32mQuestion types predicted with bert:\n{bert_qtype}\033[0m")
    logging.info(f"Question types predicted with bert:\n{bert_qtype}")

    unmatched_df = pd.DataFrame(wrong, 
                      columns=['Para No.', 'Question No.', 'Question', 'Actual Ans', 'Predicted Ans'])
    logging.info(f'Wrong Dataframe:\n{unmatched_df}\n')
    print('\033[1;32mOutput analysed successfully\033[0m')

    return unmatched_df

def main():
    df, all_sent, _, qtypes, mcq, finalmcq, finalpara, adjustedIndex, qns, questions = load_required_files_and_display_data()
    # Load the model Model
    embedder1 = "bert-base-nli-mean-tokens"
    embedder2 = "all-mpnet-base-v2"
    embedder3 = "all-distilroberta-v1"

    choice = int(input('Enter:\n\'0\' for the all-MiniLM-L6-v2 embedder\
    \n\'1\' for the all-mpnet-base-v2 embedder\
    \n\'2\' for the all-distilroberta-v1 embedder\
    \n\'3\' for another embedder you want\
    \nEnter your choice: '))
    match choice:
        case 0: embedder_name = embedder1
        case 1: embedder_name = embedder2
        case 2: embedder_name = embedder3
        case 3: embedder_name = input('Copy-paste the name of the embedder: ')
        case _: print('Please enter a valid choice')
    
    embedder = SentenceTransformer(embedder_name)
    # model = SentenceTransformer('bert_base_nli_mean_tokens')

    start_time = time.time()

    edge_output_dict = similarity_scoring_using_edge_matching(mcq, finalmcq, finalpara, adjustedIndex)

    print(f'\033[0;32mGetting Corpus for Index 1...\n{getCorpus(1, df, embedder)}\033[0m')
    logging.info(f'getCorpus(1):\n{getCorpus(1, df, embedder)}\n')

    print(f'\033[0;32mGetting Score for Index 1...\n{getScore(1, embedder, qns, df)}\033[0m')
    logging.info(f'getScore(1):\n{getScore(1, embedder, qns, df)}\n')

    print(f'\033[0;32mGetting Score for Index 4...\n{getScore(4, embedder, qns, df)}\033[0m')
    logging.info(f'getScore(4):\n{getScore(4, embedder, qns, df)}\n')

    print(f'\033[0;32mGetting Score for Index 3...\n{getScore(3, embedder, qns, df)}\033[0m')
    logging.info(f'getScore(3):\n{getScore(3, embedder, qns, df)}\n')

    print(f'\033[0;33mDefining the output for each question...\033[0m')
    output = []

    for i in range(1, len(qns)+1):
        output.append(getScore(i, embedder, qns, df))

    print(f'\033[0;32m{output}\033[0m')
    logging.info(f'output:\n{output}\n')

    print(f'\033[0;32mGetting Output for Index 0...\n{output[0]}\033[0m')
    logging.info(f'output[0]:\n{output[0]}\n')

    print(f'\033[0;32mGetting Output for Index 1...\n{output[1]}\033[0m')
    logging.info(f'output[1]:\n{output[1]}\n')

    print(f'\033[0;32mGetting Output for Index 2...\n{output[2]}\033[0m')
    logging.info(f'output[2]:\n{output[2]}\n')

    print(f'\033[0;32mGetting Output for Index 3...\n{output[3]}\033[0m')
    logging.info(f'output[3]:\n{output[3]}\n')

    sbert_and_edge_output_dict = get_score_from_sbert_and_edge_matching(output, edge_output_dict["edge_score"],
                                                                        qns, qtypes, edge_output_dict["pred_dict"])
    
    unmatched_df = analyze_output(sbert_and_edge_output_dict["corr_qna"], sbert_and_edge_output_dict["incorr_qna"],
                                  sbert_and_edge_output_dict["bert_match"], sbert_and_edge_output_dict["bert_qtype"],
                                  sbert_and_edge_output_dict["unmatched"], sbert_and_edge_output_dict["wrong"])
    
    end_time = time.time()

    logging.info(f'''================================================================================
                \n================================================================================
                \n================================================================================
                \n==============================   RUN SUMMARY   =================================
                \n================================================================================
                \n================================================================================
                \n================================================================================
                \n                     MODEL NAME: {embedder_name}
                \n                      Total number of questions: {len(all_sent)}
                \n                Number of questions predicted correctly: {len(sbert_and_edge_output_dict["corr_qna"])}
                \n               Number of questions predicted incorrectly: {len(sbert_and_edge_output_dict["unmatched"])}
                \n                        ACCURACY: {(len(sbert_and_edge_output_dict["corr_qna"]) / len(all_sent)) * 100}
                \n                            ERROR: {(len(sbert_and_edge_output_dict["unmatched"]) / len(all_sent)) * 100}
                \n                          RUNTIME: {end_time - start_time}s
                \n================================================================================
                \n================================================================================
                \n================================================================================
                \n================================================================================
                \n================================================================================
                \n================================================================================
                \n================================================================================''')

    IST = pytz.timezone('Asia/Kolkata')
    ist_now = datetime.now(IST)
    date = ist_now.strftime('%d-%m-%Y')
    cur_time = ist_now.strftime('%H-%M-%S')
    logging.info(f'END OF LOGS FOR THE RUN PERFORMED ON {date}, {cur_time}\n \
    The file is saved as \
    "{date}_{cur_time}___310Sci-Wrong_Ans_from_4_Sbert+edge___sci4_reorder_edge_plus_SBert-310___{embedder_name}.xlsx"')

    # Save the Excel Sheet of Incorrectly Matched Questions
    # Format: Datetime___Filename___Codename___Modelname.xlsx
    unmatched_df.to_excel(f"{date}_{cur_time}___310Sci-Wrong_Ans_from_4_Sbert+edge___sci4_reorder_edge_plus_SBert-310___{embedder_name}.xlsx")

    try:
        shutil.copy(f"{date}_{cur_time}___310Sci-Wrong_Ans_from_4_Sbert+edge___sci4_reorder_edge_plus_SBert-310___{embedder_name}.xlsx",
                    "./data/incorrect_matches")
        os.remove(f"./{date}_{cur_time}___310Sci-Wrong_Ans_from_4_Sbert+edge___sci4_reorder_edge_plus_SBert-310___{embedder_name}.xlsx")
    except Exception as e:
        print(f'An error occurred while trying to save the errors excel file:\n{e}')
    
    # Save the logfile of outputs for future reference and/or debugging
    # Format: Datetime___Codename___Modelname___logfile.log
    try:
        # Shut down logging and release the log file
        logging.shutdown()
        # Give time for the process to complete
        time.sleep(0.5)
        # Copy log file to desired directory with desired name
        shutil.copy('./log_file.log',
                    f'./logs/{date}_{cur_time}___sci4_reorder_edge_plus_SBert-310___{embedder_name}___logfile.log')
        
        os.remove('./log_file.log')
    except Exception as e:
        print(f'An error occurred while trying to save the logs file:\n{e}') 

    print(f'''\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m==============================   RUN SUMMARY   =================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1;33m                         MODEL NAME: {embedder_name}\033[0m
            \n\033[1;33m                       Total number of questions: {len(all_sent)}\033[0m
            \n\033[1;32m                Number of questions predicted correctly: {len(sbert_and_edge_output_dict["corr_qna"])}\033[0m
            \n\033[1;31m                Number of questions predicted incorrectly: {len(sbert_and_edge_output_dict["unmatched"])}\033[0m
            \n\033[1;32m                            ACCURACY: {(len(sbert_and_edge_output_dict["corr_qna"]) / len(all_sent)) * 100}\033[0m
            \n\033[1;31m                             ERROR: {(len(sbert_and_edge_output_dict["unmatched"]) / len(all_sent)) * 100}\033[0m
            \n\033[1;33m                             RUNTIME: {end_time - start_time}s                  \033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m
            \n\033[1m================================================================================\033[0m''')

if __name__ == "__main__":
    main()