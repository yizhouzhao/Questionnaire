#generate questions
import requests
import spacy

nlp =  spacy.load("en_core_web_sm")

def generate_noun_question_good_or_bad(word, good_or_bad = "good"):
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

def generate_one_question(word, word_pos, noun_question_type = "exist"):
    '''
    Generate one question from the key word
    :params:
        trigger: key word
        noun_question_type: exist: is there any .... "good or bad": is the text saying anything good/bad about....
    :return:
        a question
    '''
    if word_pos in ["VERB", "verb", "v"]:
        return "is anybody {} mentioned in the text?".format(word)
    elif word_pos in ["ADJ", "adj"]:
        return "is there anything {} mentioned in the text?".format(word) 
    else: #noun
        if noun_question_type == "exist":
            return "is anything related to {} mentioned in the text?".format(word)
        else: #noun_question_type == "good or bad":
            return generate_noun_question_good_or_bad(word, noun_question_type)

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
            obj["word"] = obj["word"].lower()
            if not trigger in obj["word"]: #if is meaningful
                if 'adj' in obj['tags']:
                    one_question = generate_one_question(obj['word'], "adj")
                    question_list.append(one_question)
                    #question_list.append("is there anything {} mentioned in the text?".format(obj['word']))
                elif 'n' in obj['tags']:
                    if question_type == "exist":
                        one_question = generate_one_question(obj['word'], "n", "exist")
                        question_list.append(one_question)
                    else:#question_type == "good or bad":
                        one_question = generate_one_question(obj['word'], "n", "good")
                        question_list.append(one_question)
                        one_question = generate_one_question(obj['word'], "n", "bad")
                        question_list.append(one_question)
                elif 'v' in obj['tags']:
                    one_question = generate_one_question(obj['word'], "v")
                    question_list.append(one_question)
                
                if len(question_list) >= max_question_num:
                    break
    elif source == "conceptnet":
        assert appendix is None
        url_link = "http://api.conceptnet.io/related/c/en/{}?filter=/c/en".format(trigger)
        print("generate questions from link: ", url_link)
        objects = requests.get(url_link).json()['related']
        for obj in objects:
            related_word = obj['@id'].split('/')[-1].replace("_", " ").lower()
            if not trigger in related_word.startswith: #if is meaningful
                if question_type == "exist":
                    question_list.append("is anything related to {} mentioned in the text?".format(related_word))
                else:#question_type == "good or bad":
                    question_list.append(generate_noun_question_good_or_bad(related_word, "good"))
                    question_list.append(generate_noun_question_good_or_bad(related_word, "bad"))
            if len(question_list) >= max_question_num:
                break
    
    return question_list