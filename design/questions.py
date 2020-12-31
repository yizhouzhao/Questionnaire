#generate questions
import requests
import spacy

nlp =  spacy.load("en_core_web_sm")

def question_good_or_bad(word, good_or_bad = "good"):
    '''
    Generate a question about a word saying good or bad 
    '''
    lemma_tags = {"NNS", "NNPS"}
    vowels = {"a", "e", "i", "o", "u"}
    token = nlp(word)[0]
    if token.tag_ in lemma_tags:
        return "is the text saying anything {} about {}?".format(good_or_bad, word)
    else:
        if word[0] in vowels:
            return "is the text saying anything {} about an {}?".format(good_or_bad, word)
        else:
            return "is the text saying anything {} about a {}?".format(good_or_bad, word)

def generate_one_question(word, noun_question_type = "exist"):
    '''
    Generate one question from the key word
    :params:
        trigger: key word
        noun_question_type: exist: is there any .... "good or bad": is the text saying anything good/bad about....
    :return:
        a question
    '''
    token = nlp(word)[0]



def generate_questions(trigger:str, source="datamuse", appendix:list=None, max_question_num=10, question_type="exist"):
    '''
    Generate questions from key words
    :params:
        trigger: key word
        source: datamuse, glove, or conceptnet
        appendix: List[str] of a additional keywords
        max_question_num: max number of questions to be generated
        question_type: exist: is there any .... "good or bad": is the text saying anything good/bad about....
    :return:
        a list of questions
    '''
    question_list = []

    if source == "datamuse":
        appendix_str = "+".join(appendix) if appendix != None else ""
        url_link = "https://api.datamuse.com/words?rel_trg={}&topics={}&md=p".format(trigger, appendix_str)
        print("generate questions from link: ", url_link)
        objects = requests.get(url_link).json()
        for obj in objects:
            if 'adj' in obj['tags']:
                question_list.append("is there anything {} mentioned in the text?".format(obj['word']))
            elif 'n' in obj['tags']:
                if question_type == "exist":
                    question_list.append("is anything related to {} mentioned in the text?".format(obj['word']))
                else:#question_type == "good or bad":
                    question_list.append(question_good_or_bad(obj['word'], "good"))
                    question_list.append(question_good_or_bad(obj['word'], "bad"))
            elif 'v' in obj['tags']:
                question_list.append("is {} mentioned in the text?".format(obj['word'])) #病句
            
            if len(question_list) >= max_question_num:
                break
    elif source == "conceptnet":
        assert appendix is None
        url_link = "http://api.conceptnet.io/related/c/en/{}?filter=/c/en".format(trigger)
        print("generate questions from link: ", url_link)
        objects = requests.get(url_link).json()['related']
        for obj in objects:
            related_word = obj['@id'].split('/')[-1]
            if related_word in nlp.vocab and (not related_word.startswith(trigger)): #if is meaningful
                if question_type == "exist":
                    question_list.append("is anything related to {} mentioned in the text?".format(obj['word']))
                else:#question_type == "good or bad":
                    question_list.append(question_good_or_bad(obj['word'], "good"))
                    question_list.append(question_good_or_bad(obj['word'], "bad"))
            if len(question_list) >= max_question_num:
                break
    
    return question_list