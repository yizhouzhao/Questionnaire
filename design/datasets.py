# model
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .util import generate_answer_text, generate_unifiedqa_text
from .preprocess import *

class QuestionCollection(object):
    '''
    Collection of questions
    :params
        question_file: question file path, csv format
    '''
    def __init__(self, question_file:str):
        #load question
        self.question_file = question_file
        self.question_answer_list = self.load_questions()

    def load_questions(self):
        '''
        load questions
        :return:
            a list containing questions and answers
        '''
        question_answer_list = []
        df = pd.read_csv(self.question_file)
        for i in range(len(df)):
            question_type = df.iloc[i][0]
            question = df.iloc[i][1]
            if question_type == "Multiple-choice":
                answer = generate_answer_text(df.iloc[i][2].split(","), add_change_line=False)
            else: #question_type == "Yes-no":
                answer = generate_answer_text(["yes","no"], add_change_line=False)

            question_answer_list.append([question.lower(), answer.lower()])

        return question_answer_list
    

class QAMachine(object):
    '''
    A machine to hold dataset and qa-model to perform question and answering
    '''
    def __init__(self, question_collection_file:str, dataset_name:str, model_name:str="allenai/unifiedqa-t5-large"):
        '''
        :params:
            question_collection_file: the name of the question file
            dataset_name: the name of the dataset listed in #from datasets import list_datasets
            model_name: the name of the model in UnifiedQA
        '''
        self.dataset_name = dataset_name
        self.model_name = model_name

        #load question collection
        self.question_collection = QuestionCollection(question_collection_file)

        # load data
        self.dataset = [] #dataset
        self.label_set = [] #all kinds of labels
        self.load_dataset()

        # load model
        self.model = None
        self.tokenizer = None
        self.load_model()

        # survey
        self.survey = np.zeros(shape=(len(self.dataset), len(self.question_collection.question_answer_list)))

    def load_dataset(self, split="train"):
        '''
        Load dataset
        :params:
            split: train or test
        :return:
        '''
        self.dataset = load_dataset(self.dataset_name, split=split)
        self.preprocess_dataset()
        self.label_set = set([item["label"] for item in self.dataset])

    def preprocess_dataset(self):
        if self.dataset_name == "ag_news":
            AG_NEWS(self.dataset)

    def load_model(self):
        print("load model......")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
        res = self.model.generate(input_ids, **generator_args)
        return [self.tokenizer.decode(x) for x in res]

    def conduct_survey(self, question_id_list:list):
        '''
        running Q&A on dataset
        '''
        for question_id in question_id_list:
            print("Datasets QAMachine conduct survey on question {} : {}".format(str(question_id), 
                self.question_collection.question_answer_list[question_id][0]))
            for i in range(len(self.dataset)):
                pass







            


            




