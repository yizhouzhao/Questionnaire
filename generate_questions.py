import pandas as pd

from design.questions import *
from questions.prior_questions import *

from argparse import ArgumentParser

def make_args():
    parser = ArgumentParser()
    #dataset 
    parser.add_argument('--dataset', dest='dataset', default='ag_news', type=str, help='dataset name')

    return parser.parse_args("")

if __name__ == '__main__':
    args = make_args()
    #generate questions from keywords
    if args.dataset == "ag_news":
        keywords = g_ag_news_keywords
        noun_question_type = "exist"
        question_save_file_name = "agnews"
    elif args.dataset == "imdb" or  args.dataset == "rotten_tomatoes":
        keywords = g_movie_piror_keywords
        noun_question_type = "good or bad"
        question_save_file_name = "movie"
    else:
        pass

    question_save_file = "questions/{}questions.csv".format(question_save_file_name)

    questions = []
    for i in range(len(keywords)):
        word = keywords[i]
        if noun_question_type == "exist":
            questions.append(generate_one_question(word, "n", noun_question_type="exist"))
            questions.extend(generate_questions(word, source="datamuse", appendix=None, max_question_num=20, question_type="exist"))
        else: #noun_question_type == "good or bad":
            questions.append(generate_one_question(word, "n", noun_question_type="good"))
            questions.append(generate_one_question(word, "n", noun_question_type="bad"))
            questions.extend(generate_questions(word, source="datamuse", appendix=None, max_question_num=20, question_type="good or bad"))

    question_types = ["Yes-no" for _ in range(len(questions))]
    questionnaire_data = pd.DataFrame({"Type":question_types, "Question": questions})


    questionnaire_data.to_csv(question_save_file, index=False)