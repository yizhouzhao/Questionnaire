import pandas as pd

from design.questions import *
from questions.prior_questions import *

if __name__ == '__main__':
    #generate questions from keywords
    keywords = g_movie_piror_keywords
    question_save_file = "questions/moviequestions.csv"

    questions = []
    for i in range(len(keywords)):
        word = keywords[i]
        questions.append(generate_one_question(word, "n", "good"))
        questions.append(generate_one_question(word, "n", "bad"))
        #questions.extend(generate_questions(word, source="datamuse", appendix=None, max_question_num=20, question_type="good or bad"))

    question_types = ["Yes-no" for _ in range(len(questions))]
    questionnaire_data = pd.DataFrame({"Type":question_types, "Question": questions})


    questionnaire_data.to_csv(question_save_file, index=False)