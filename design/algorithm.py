#algorithm
# model
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from fuzzywuzzy import process

from .util import generate_answer_text, generate_unifiedqa_text

from .aot import *

class AOTMachine():
    def __init__(self, aot:AndOrTree,
                model_name:str="allenai/unifiedqa-t5-large",
                batch_size = 32, gpu_index=0):
        # property
        # self.dataset_name = dataset_name
        self.model_name = model_name

        #aot
        self.aot = aot

        # load data
        # self.train_dataset = self.load_dataset(split="train") #dataset
        # self.test_dataset = self.load_dataset(split="test") #dataset

        # device
        self.use_cuda = True and torch.cuda.is_available()
        self.device = torch.device("cuda:{}".format(gpu_index) if self.use_cuda else "cpu")

        # load model
        self.model = None
        self.tokenizer = None
        self.load_model()
        self.batch_size = batch_size #batch size to run the QA model

    # def load_dataset(self, split="train"):
    #     '''
    #     Load dataset
    #     :params:
    #         split: train or test
    #     :return:
    #     '''
    #     dataset = load_dataset(self.dataset_name, split=split)
    #     #self.preprocess_dataset()
    #     #label_set = set([item["label"] for item in dataset])
    #
    #     return dataset

    def load_model(self):
        print("load model......")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        if self.use_cuda:
            self.model = self.model.to(self.device)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer(input_string, padding=True, truncation=True, max_length=100, return_tensors="pt").input_ids
        if self.use_cuda:
            input_ids = input_ids.to(self.device)
        res = self.model.generate(input_ids, **generator_args)
        return [self.tokenizer.decode(x) for x in res]

    def conduct_survey_along_samples_and_node(self, samples, node:TreeNode):
        '''
        running Q&A on dataset on questions(question_id_list)
        '''

        # print("Datasets QAMachine conduct survey on question {} : {}".format(str(question_id),
        #    self.question_collection.question_answer_list[question_id][0]))
        survey = np.zeros((len(samples), len(node.questions)))
        for i in range(len(node.questions)):
            for j in tqdm(range(0, len(samples), self.batch_size)):
                batch_sentences = []
                for k in range(j, min(self.batch_size + j, len(samples))):
                    answer = generate_answer_text(node.answers, add_change_line=False)

                    text = generate_unifiedqa_text(node.questions[i],
                                                   answer,
                                                   samples[k]['text'])

                    batch_sentences.append(text)

                question_answers = self.run_model(batch_sentences)

                for k in range(len(question_answers)):
                    answer_choice = process.extractOne(question_answers[k],
                                                       node.answers)[0]
                    answer_index = node.answers.index(answer_choice)

                    survey[j + k][i] = 1.0 if answer_index < 1 else -1.0

        return survey