# model
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from fuzzywuzzy import process

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
        self.question_list = [] # question in text
        self.answer_list = [] # answer in text

        self.raw_answer_list = [] # answer in choice
        self.load_questions()

    def load_questions(self):
        '''
        load questions
        :return:
            a list containing questions and answers
        '''
        df = pd.read_csv(self.question_file)
        for i in range(len(df)):
            question_type = df.iloc[i][0]
            question = df.iloc[i][1]
            if question_type == "Multiple-choice":
                raw_answer = df.iloc[i][2].split(",")
            else: #question_type == "Yes-no":
                raw_answer = ["yes","no"]
            
            self.raw_answer_list.append(raw_answer)

            answer = generate_answer_text(raw_answer, add_change_line=False)
            self.question_list.append(question.lower())
            self.answer_list.append(answer.lower())

    def __len__(self):
        return len(self.question_list)
    

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
        self.survey = np.zeros(shape=(len(self.dataset), len(self.question_collection)))

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
        
        #print("Datasets QAMachine conduct survey on question {} : {}".format(str(question_id), 
        #    self.question_collection.question_answer_list[question_id][0]))
        for i in tqdm(range(len(self.dataset))):
            for question_id in question_id_list:
                text = generate_unifiedqa_text(self.question_collection.question_list[question_id], 
                            self.question_collection.answer_list[question_id], 
                            self.dataset[i]['text'])
                question_answer = self.run_model(text)[0]
                answer_choice = process.extractOne(question_answer, self.question_collection.raw_answer_list[question_id])[0]
                answer_index = self.question_collection.raw_answer_list[question_id].index(answer_choice)
                print("Datasets QAMachine conduct survey", text, question_answer, answer_choice, answer_index)
                self.survey[i][question_id] = 1.0 if answer_index < 1 else -1.0
                
    def save_survey(self, survey_file_path):
        np.savetxt("survey.csv", self.survey, delimiter=",")






            


            




