#help functions

def generate_answer_text(option_list:list, add_change_line=True):
    '''
    Generate anaswer text
    :param option_list: a list of options
    :return: a string of answers in UnifiedQA style
    '''
    answer_text = ""
    for i, option in enumerate(option_list):
        char_code = 65 + i
        answer_text += "({}) {} ".format(chr(char_code), option)
    if add_change_line:
        answer_text += "\n "
    return answer_text

def generate_unifiedqa_text(question:str, answer:str, description:str):
    '''
    Generate text that can be fed into unifiedqa model
    :parmas
        question: 
        answer:
        description:
    :return
        text string
    '''
    qa_string = "{} \n {} \n {}".format(question, answer, description)
    return qa_string.lower()

