import argparse
from design.datasets import *

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='generate survey data')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                 help='an integer for the accumulator')
    QAM = QAMachine("design/questions.csv", "ag_news")
    QAM.conduct_survey(question_id_list=[j for j in range(len(QAM.question_collection))])
    QAM.save_survey("record/survey_{}_basic_questions.csv".format(QAM.dataset_name))
