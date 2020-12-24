from argparse import ArgumentParser
from design.datasets import *

from datetime import datetime

def make_args():
    parser = ArgumentParser()

    #dataset 
    parser.add_argument('--dataset', dest='dataset', default='ag_news', type=str, help='dataset name')
    parser.add_argument('--dataset_split', dest="dataset_split", default="train", type=str, help='dataset split: train or test')
    parser.add_argument('--questionnaire', dest='questionnaire', default="basicquestions", type=str, help="questionnaire path")
    parser.add_argument('--unifiedqa_model', dest="unifiedqa_model", default="unifiedqa-t5-large", type=str, help='qa model name')
    

    return parser.parse_args()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='generate survey data')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                 help='an integer for the accumulator')
    args = make_args()
    questionnaire_path = "questions/{}.csv".format(args.questionnaire)
    qa_model_name = "allenai/{}".format(args.unifiedqa_model)

    #print time
    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("Start Time: ", current_time)

    QAM = QAMachine(question_collection_file=questionnaire_path, dataset_name=args.dataset, model_name=qa_model_name, dataset_split=args.dataset_split)
    QAM.conduct_survey(question_id_list=[j for j in range(len(QAM.question_collection))])
    QAM.save_survey("record/survey_{}({})_{}_{}.csv".format(args.dataset, args.dataset_split, args.questionnaire, args.unifiedqa_model))

    #print time
    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y %H:%M:%S")
    print("End Time: ", current_time)