import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import string
import ast

import logging
import os
import sys
import shutil
from datetime import date, datetime
import pytz
import time

def load_required_files_and_display_data():
    # Load the sci310_Top-Sents_Set-per-Question.csv file
    # qno,Questions,new_paratext
    set_per_question = "270824_sci310_Top-Sents_Set-per-Question.csv"
    print(f'Loading the {set_per_question} file...')
    df = pd.read_csv(f"./data/{set_per_question}")

    # Load the sci310_QuestionType.csv file
    # No.,qno,Ans1,Ans2,Ans3,Ans4,Correct Ans,Question Type,Question Sentence
    question_type = "270824_sci310_QuestionType.csv"
    print(f'Loading the {question_type} file...')
    qtype_df = pd.read_csv(f"./data/{question_type}")

    # Set up the log file
    logging.basicConfig(filename='log_file.log',
                        level=logging.INFO,
                        filemode='a',
                        force=True,
                    )
    # a million thanks to https://stackoverflow.com/questions/54597462/problem-with-logging-module-in-google-colab

    # Get information about the type of questions in the dataset
    qtypes, _, _ = question_information(qtype_df)

    # Get preview of the df DataFrame
    print(f'Printing a preview of the \'df\' DataFrame which holds the data of the {set_per_question} file...')
    print(df.head())

    logging.info(f'df.head():\n{df.head()}\n')

    print(f'Printing the list of the first row without AST...')
    print(df["new_paratext"][0])

    logging.info(f"df['new_paratext'][0]:\n{df['new_paratext'][0]}\n")

    print(f'Printing the list of the first row with AST...')
    print(ast.literal_eval(df['new_paratext'][0]))

    logging.info(f"ast.literal_eval(df['new_paratext'][0]):\n{ast.literal_eval(df['new_paratext'][0])}\n")

    # qno, Question, Ans1, Ans2, Ans3, Ans4, support and correct_answer columns
    sci_para_mcqframed = "270824_Sci-Para-mcqframed-Data-310.xlsx"
    print(f'Loading and checking the {sci_para_mcqframed} file...')
    qns = pd.read_excel(f'./data/{sci_para_mcqframed}')

    print('Printing the head of the DataFrame...')
    print(qns.head())
    logging.info(f"qns.head():\n{qns.head()}\n")

    print('Printing the first row of the DataFrame...')
    print(qns.loc[[0]])
    logging.info(f"qns.loc[[0]]:\n{qns.loc[[0]]}\n")

    # Declare the list of questions variable
    questions = []

    for index, row in qns.iterrows():
        questions.append(row['Question'])

    print(f'The complete list of questions in {sci_para_mcqframed} file are:\n')
    print(questions)

    logging.info(f'The complete list of questions are:\n{questions}\n')

    return df, qtype_df, qtypes, qns, questions

def question_information(qtype_df):
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

    print(f'Number of \'which\' questions: {count_dict_all_categories["which"]}')
    print(f'Number of \'what\' questions: {count_dict_all_categories["what"]}')
    print(f'Number of \'how\' questions: {count_dict_all_categories["how"]}')
    print(f'Number of \'why\' questions: {count_dict_all_categories["why"]}')
    print(f'Number of \'where\' questions: {count_dict_all_categories["where"]}')
    print(f'Number of \'who\' questions: {count_dict_all_categories["who"]}')
    print(f'Number of \'when\' questions: {count_dict_all_categories["when"]}')
    print(f'Number of \'other\' questions: {count_dict_all_categories["where"] + count_dict_all_categories["who"] + count_dict_all_categories["when"]}')

    return qtypes, count_dict, count_dict_all_categories

def getFramedOpts(qns, quesn: int) -> list:
    """
    Retrieve and format the MCQ Answer Options (Ans1, Ans2, Ans3, Ans4)
    for a question from the qns dataframe.

    Note:
        Assumes that quesn is 1-based index (first question at index 1)
        Removes newline characters

    Arguments:
        quesn (int): Index of the question being considered

    Returns:
        framedoptions (list): A list of the answer options for the question
    """
    framedoptions = []
    framedoptions.append(qns.iloc[quesn-1]['Ans1'].replace('\n', ''))
    framedoptions.append(qns.iloc[quesn-1]['Ans2'].replace('\n', ''))
    framedoptions.append(qns.iloc[quesn-1]['Ans3'].replace('\n', ''))
    framedoptions.append(qns.iloc[quesn-1]['Ans4'].replace('\n', ''))
    return framedoptions

# To generate corpus and corpus embeddings for a particular question no.
def getCorpus(embedder, df, quesn: int) -> tuple:
    """
    Generate and process the text data related to the quesn question from the
    dataset. Compute its embeddings as tensors.

    Note:
        Assumes that quesn is 1-based (first question starts at index 1)
        Uses ast.literal_eval() to safely convert the string and convert it
        into a Python list of sentences.
        Uses the provided model for encoding the corpus of sentences.

    Arguments:
        quesn (int): The index of the question under consideration

    Returns:
        A tuple containing:
            corpus_embeddings (torch.Tensor): Tensor embeddings of the list of
            sentences
            corpus (list): The cleaned list of sentences
    """
    corpus = ast.literal_eval(df['new_paratext'][quesn-1])
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    return corpus_embeddings, corpus

def getScore(embedder, qns, df, quesn: int) -> list:
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
    
    Requires:
        getFramedOpts(qns, quesn) to get the framed options for a question
        getCorpus(embedder, df, quesn) to get the corpus of supporting sentences and
        to get the embeddings of the corpus for a question

    Arguments:
        quesn (int): The index of the question in consideration

    Returns:
        final_answer (list): A list of tuples. Each tuple consists of
        an answer, the corresponding most similar supporting sentence
        and the similarity score
    """
    framedOptions = getFramedOpts(qns, quesn)
    corpus_embeddings, corpus = getCorpus(embedder, df, quesn)

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

    # print(f'The total number of questions is {len(qns)}')
    # logging.info(f'The total number of questions is {len(qns)}\n')

    return final_answer

def define_the_output(embedder, qns, df):
    """
    This function gets the score of each answer option (Score of Ans1, Score of Ans2,
    Score of Ans3, Score of Ans4) for each question (Q1, ..., Q310). The score of an
    answer is the Cosine Similarity score of that answer with the most similar supporting
    sentence from the corpus of supporting sentences for that question.

    Requires:
        getScore(embedder, qns, quesn) to perform the task described above
    """
    output = []
    for i in range(1, len(qns)+1):
        output.append(getScore(embedder, qns, df, i))

    print(output)
    logging.info(f"output:\n{output}\n")

    print('The first list of the output variable, tells for each possible answer\
          (Ans1, Ans2, Ans3, Ans4) of the first question (Q1) the most similar supporting\
          sentence as well as the cosine similarity score with that supporting sentence.')
    print(output[0])
    logging.info(f"output[0]:\n{output[0]}\n")

    print('The second list of the output variable, tells for each possible answer\
          (Ans1, Ans2, Ans3, Ans4) of the second question (Q2) the most similar supporting\
          sentence as well as the cosine similarity score with that supporting sentence.')
    print(output[1])
    logging.info(f"output[1]:\n{output[1]}\n")

    print('The fifteenth list of the output variable, tells for each possible answer\
          (Ans1, Ans2, Ans3, Ans4) of the fifteenth question (Q15) the most similar supporting\
          sentence as well as the cosine similarity score with that supporting sentence.')
    print(output[14])
    logging.info(f"output[14]:\n{output[14]}\n")

    return output

def getbertScore(scoreList: list) -> tuple:
    '''
    This function takes a list of tuples. Each tuple tells about an
    answer to a question, the most matching supporting sentence, and
    the score of similarity. The answer with the highest score and the
    score is returned.

    Arguments:
        scoreList (list): A list of tuples of the format
        [(answer, matching sentence, similarity score), ...]

    Returns:
        A tuple containing:
            maxx[0], the answer which has the highest score
            maxx[2], the score of that answer
    '''
    maxx = (-1, -1, -1)
    for tup in scoreList:
        if float(tup[2]) > float(maxx[2]):
            maxx = tup
    # Return the Most Relevant Answer and the Corresponding Score
    return (maxx[0], maxx[2])

def removepun(stringinp: str) -> str:
    """
    Remove punctuation marks and other special characters and
    escape sequences from text in order to avoid scoring errors.

    Arguments:
        stringinp (str): The text to be cleaned

    Returns:
        stringinp (str): The cleaned text
    """
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

def get_score_from_sbert(qns, qtypes, output):
    """
    This function is used to get the score from SBert.
    """
    # List of indices of incorrectly matched questions, i.e., questions whose
    # predicted as well as correct answer do not match.
    unmatched=[]
    # List of indices of correctly matched questions, i.e., questions whose
    # predicted as well as correct answer matches.
    bert_match=[]
    # Dictionary wherein the keys are the question types and the value is the
    # number of occurrences of each question type.
    bert_qtype={}
    # List of indices of correctly matched questions, i.e., questions whose
    # predicted as well as correct answer matches.
    matchedques=[]
    # list of the index of the correctly matched question, the question itself
    # as well as the correct answer
    corr_qna=[]
    # A list of tuples of index of the question, the question itself and the
    # correct answer, for the unmatched questions.
    incorr_qna=[]
    # Dictionary wherein the keys are the question types and the value is the
    # number of occurrences of each question type.
    pred_dict={}
    # This is a dataframe of rows. Each row is a list of values: the index of
    # the question, the question itelf, the correct answer, the predicted answer
    # and the score. This is updated for each incorrectly matched question.
    wrong_df=[]

    # @title Get Score
    for i in range(len(output)):
        if (i+1,1) not in matchedques:
            print('----------------------------------------------------------------')
            logging.info('----------------------------------------------------------------\n')
            pred_opt, score = getbertScore(output[i])
            print(f'\033[1;32mPredicted Output: {pred_opt}\033[0m')
            logging.info(f'pred_opt:\n{pred_opt}\n')
            print(f"\033[1;32mCorrect Answer: {qns['correct_answer'][i]}\033[0m")
            logging.info(f"qns['correct_answer'][i]:\n{qns['correct_answer'][i]}\n")
            cleaned_predicted_output = removepun(str(pred_opt).lower())
            cleaned_correct_answer = removepun(str(qns['correct_answer'][i]).lower())
            if cleaned_predicted_output == cleaned_correct_answer:
                matchedques.append((int(i + 1), 1))
                corr_qna.append((int(i + 1), qns["Question"][i], qns["correct_answer"][i]))
                bert_match.append((i + 1, 1))
                bert_qtype.update({qtypes[i] : bert_qtype.get(qtypes[i], 0) + 1})
                pred_dict.update({qtypes[i] : pred_dict.get(qtypes[i], 0) + 1})
            else:
                wrong_df.append([int(i + 1), qns["Question"][i], qns["correct_answer"][i], str(pred_opt), score])
                unmatched.append((int(i + 1), 1))
                incorr_qna.append((int(i + 1), qns["Question"][i], qns["correct_answer"][i]))

    return {"matchedques": matchedques,
            "corr_qna": corr_qna,
            "bert_match": bert_match,
            "bert_qtype": bert_qtype,
            "pred_dict": pred_dict,
            "wrong_df": wrong_df,
            "unmatched": unmatched,
            "incorr_qna": incorr_qna}

def analyze_output(matchedques, corr_qna, unmatched, incorr_qna, bert_qtype):
    """
    This function is used to analyse the output.
    """
    print(f"Total correctly predicted by Bert: {len(matchedques)}")
    logging.info(f"Total correctly predicted by Bert: {len(matchedques)}\n")
    # Here, this match is more than 71, if correct pair of predicted answer and actual answer is matched.
    # get this number for other two models
    # duplicate the code

    print(f'Correctly matched questions are:\n{matchedques}')
    logging.info(f'matchedques:\n{matchedques}\n')

    print("\033[1;32m--- Correct MCQs ---\nQuestion No. || Question ==> Correct Ans.\n\033[0m")
    logging.info("--- Correct MCQs ---\nQuestion No. || Question ==> Correct Ans.\n")
    for i in corr_qna:
        print("\033[1;32m", i[0], "||", i[1], "==>", i[2], "\033[0m")
        logging.info(f"\n{i[0]} \|\| {i[1]} ==> {i[2]}\n")

    print(f'Incorrectly matched questions are:\n{unmatched}')
    logging.info(f"Questions not matched\n{unmatched}\n")

    print("\033[1;31m--- Incorrect MCQs ---\nQuestion No. || Question ==> Correct Ans.\033[0m")
    logging.info("--- Incorrect MCQs ---\nQuestion No. || Question ==> Correct Ans.\n")
    for i in incorr_qna:
        print("\033[1;31m", i[0], "||", i[1], "==>", i[2], "\033[0m")
        logging.info(f"{i[0]} \|\| {i[1]} ==> {i[2]} \n")

    print(f"\n\n\n\n\n\033[1;32mQuestion types predicted with bert --- {bert_qtype}\033[0m")
    logging.info(f"\n\n\n\n\nQuestion types predicted with bert --- {bert_qtype}\n")

def main():
    df, _, qtypes, qns, _ = load_required_files_and_display_data()

    # Load the model Model
    model1 = "all-MiniLM-L6-v2"
    model2 = "all-mpnet-base-v2"
    model3 = "all-distilroberta-v1"

    choice = int(input('Enter:\n\'0\' for the all-MiniLM-L6-v2 model\
    \n\'1\' for the all-mpnet-base-v2 model\
    \n\'2\' for the all-distilroberta-v1 model\
    \n\'3\' for another model you want\
    \nEnter your choice: '))
    match choice:
        case 0: model = model1
        case 1: model = model2
        case 2: model = model3
        case 3: model = input('Copy-paste the name of the model: ')
        case _: print('Please enter a valid choice')
    
    embedder = SentenceTransformer(model)

    start_time = time.time()

    print('Getting the Corpus and Corpus Embeddings for the First Question...')
    print(getCorpus(embedder, df, 1))
    logging.info(f"getCorpus(1):\n{getCorpus(embedder, df, 1)}\n")

    print('Getting the Corpus and Corpus Embeddings for the Second Question...')
    print(getCorpus(embedder, df, 2))
    logging.info(f"getCorpus(2):\n{getCorpus(embedder, df, 2)}\n")

    print('Getting Score for the Answer Options of the First Question...')
    print(getScore(embedder, qns, df, 1))
    logging.info(f"getScore(1):\n{getScore(embedder, qns, df, 1)}\n")

    print('Getting Score for the Answer Options of the Fourth Question...')
    print(getScore(embedder, qns, df, 4))
    logging.info(f"getScore(4):\n{getScore(embedder, qns, df, 4)}\n")

    print('Getting Score for the Answer Options of the Third Question...')
    print(getScore(embedder, qns, df, 3))
    logging.info(f"getScore(3):\n{getScore(embedder, qns, df, 3)}\n")

    output = define_the_output(embedder, qns, df)

    result_dict = get_score_from_sbert(qns, qtypes, output)

    analyze_output(result_dict["matchedques"], 
                   result_dict["corr_qna"], 
                   result_dict["unmatched"],
                   result_dict["incorr_qna"], 
                   result_dict["bert_qtype"])
    
    wrong_df = pd.DataFrame(result_dict["wrong_df"], 
                            columns=['Question No.', 'Question', 'Actual Ans', 'Predicted Ans', 'Score'])
    logging.info(f'Incorrectly Matched Questions\n{wrong_df}\n')

    end_time = time.time()

    logging.info(f'''================================================================================
                \n================================================================================
                \n================================================================================
                \n==============================   RUN SUMMARY   =================================
                \n================================================================================
                \n================================================================================
                \n================================================================================
                \n                         MODEL NAME: {model}
                \n                      Total number of questions: {len(qns)}
                \n                Number of questions predicted correctly: {len(result_dict["matchedques"])}
                \n               Number of questions predicted incorrectly: {len(result_dict["unmatched"])}
                \n                        ACCURACY: {(len(result_dict["matchedques"]) / len(qns)) * 100}
                \n                            ERROR: {(len(result_dict["unmatched"]) / len(qns)) * 100}
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
                 "{date}_{cur_time}___310-Sci-Wrong_Ans_from_SbertOnly___SBertOnly-sci-310___{model}.xlsx"')

    # Save the Excel Sheet of Incorrectly Matched Questions
    # Format: Datetime___Filename___Codename___Modelname.xlsx
    wrong_df.to_excel(f"{date}_{cur_time}___310-Sci-Wrong_Ans_from_SbertOnly___SBertOnly-sci-310___{model}.xlsx")

    try:
        shutil.copy(f"{date}_{cur_time}___310-Sci-Wrong_Ans_from_SbertOnly___SBertOnly-sci-310___{model}.xlsx",
                    "./data/incorrect_matches")
        os.remove(f'./{date}_{cur_time}___310-Sci-Wrong_Ans_from_SbertOnly___SBertOnly-sci-310___{model}.xlsx')
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
                    f'./logs/{date}_{cur_time}___SBertOnly-sci-310___{model}___logfile.log')
        
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
            \n\033[1;33m                         MODEL NAME: {model}\033[0m
            \n\033[1;33m                       Total number of questions: {len(qns)}\033[0m
            \n\033[1;32m                Number of questions predicted correctly: {len(result_dict["matchedques"])}\033[0m
            \n\033[1;31m                Number of questions predicted incorrectly: {len(result_dict["unmatched"])}\033[0m
            \n\033[1;32m                            ACCURACY: {(len(result_dict["matchedques"]) / len(qns)) * 100}\033[0m
            \n\033[1;31m                             ERROR: {(len(result_dict["unmatched"]) / len(qns)) * 100}\033[0m
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