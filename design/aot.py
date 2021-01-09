# design for aot
import numpy as np
import pandas as pd
from .questions import *
from .util import generate_answer_text

class TreeNode():
    '''
    And Or Tree node
    '''
    def __init__(self, keyword, part_of_speech="n", noun_question_type=None, 
                node_type="or", layer=0):
        '''
        :params:
            keyword:
            part of speech: noun adj or verb
            node type: root, and, or, terminal
        '''
        # property
        self.keyword = keyword
        self.part_of_speech = part_of_speech
        self.noun_question_type = noun_question_type
        self.node_type = node_type
        self.layer = layer

        # questionnaire
        self.questions = self.generate_questions()
        self.answers = ["yes","no"]
        self.question_types = None

        
        # tree structure
        self.parent = None
        self.children = []

        # variables
        self.visited = False 
        self.score = 1e-2 # mcts score
        self.num_simulations = 0
    
    def add_child(self, tree_node):
        self.children.append(tree_node)
        tree_node.parent = self

    def set_question(self, question:str):
        self.questions = [question]
    
    def set_answer(self, answer):
        self.answers = answer

    def __str__(self):
        return "node: " + self.keyword + " " +  self.part_of_speech + " " + self.noun_question_type + " " + self.node_type + " " + str(self.layer)

    # ----------------------------------Questionnaire----------------------------------------

    def generate_questions(self):
        '''
        Generate questions from keyword
        '''
        if (self.part_of_speech != "noun" and self.part_of_speech != "n") or self.noun_question_type == "exist":
            return [generate_one_question(self.keyword, self.part_of_speech)]
        else:

            return [generate_noun_question_good_or_bad(self.keyword, "good"),
                    generate_noun_question_good_or_bad(self.keyword, "bad")]

    # -----------------------------------MCTS------------------------------------------------

    def calculate_selection_score(self, c=np.sqrt(2))->float:
        '''
        Calculate the selection score in MCTS
        :params
            c: exploration factor in MCTS
        '''
        assert self.num_simulations > 0 and self.parent is not None and self.parent.num_simulations > 0
        return self.score / self.num_simulations + c*np.sqrt(np.log(self.parent.num_simulations) / self.num_simulations)


class AndOrTree():
    '''
    And Or Tree to hold the questionnaire
    '''
    def __init__(self, root:TreeNode):
        # property
        self.root = root
        
        # get all the question keywords
        self.keyword2node = {} 
        self.get_keyword2node()

    def get_keyword2node(self):
        '''
        Get all the question keywords
        '''
        # BFS
        node_stack = [self.root]
        while len(node_stack) > 0:
            node:TreeNode = node_stack.pop()
            
            if node.keyword not in self.keyword2node:
                self.keyword2node[node.keyword] = node
            
            for child in node.children:
                node_stack.append(child)
    
    def sample(self):
        '''
        Sample a node in this aot
        '''
        temp_node:TreeNode = self.root
        while True:
            #print(temp_node.__str__())
            if not temp_node.visited:
                return temp_node

            if len(temp_node.children) == 0:
                #if node is terminal but visited
                return None

            node_selection_index = [_ for _ in range(len(temp_node.children))]
            node_selection_probs = []
            for child in temp_node.children:
                if not child.visited:
                    node_selection_probs.append(1.0)
                else:
                    node_selection_probs.append(child.calculate_selection_score())

            probs = np.asarray(node_selection_probs) / np.sum(node_selection_probs)
            index_selected = np.random.choice(node_selection_index, p=probs)

            temp_node = temp_node.children[index_selected]

    def update_num_simulations(self, node:TreeNode):
        '''
        Update number of similations for MCTS
        '''
        temp_node = node
        temp_node.visited = True
        while temp_node is not None:
            temp_node.num_simulations += 1
            temp_node = temp_node.parent

    def tree_search(self):
        node = self.sample()
        if node is not None:
            print(node.__str__())
            #do something
            self.update_num_simulations(node)
    
def generate_subtree_from_keyword(keyword, depth, part_of_speech="n", noun_question_type="good or bad", node_type="or", layer=1, max_children=2):
    tree_node = TreeNode(keyword, part_of_speech, noun_question_type, node_type="or", layer=layer)
    if depth == 0:
        return tree_node
    else:
        trigger = keyword
        url_link = "https://api.datamuse.com/words?rel_trg={}&topics={}&md=p".format(keyword, "")
        print("generate_subtree_from_keyword {} from link: {}".format(keyword, url_link))
        objects = requests.get(url_link).json()
        for i, obj in enumerate(objects):
            if i > max_children:
                break
            obj["word"] = obj["word"].lower()
            if not trigger in obj["word"]: #if is meaningful
                child_node = generate_subtree_from_keyword(obj['word'], depth-1, obj['tags'][0], noun_question_type, node_type, layer + 1, max_children)
                tree_node.add_child(child_node)
                #question_list.append("is there anything {} mentioned in the text?".format(obj['word']))
    
        return tree_node

def generate_subtree_from_csv(csv_file:str, layer=1):
    df = pd.read_csv(csv_file)
    subtree_root_node = None
    for i in range(len(df)):
        question_type = df.iloc[i][0]
        question = df.iloc[i][1]
        if question_type == "Multiple-choice":
            raw_answer = df.iloc[i][2].split(",")
        else: #question_type == "Yes-no":
            raw_answer = ["yes","no"]
        
        #answer = generate_answer_text(raw_answer, add_change_line=False)
        keyword = df.iloc[i][3]


        if i == 0:
            tree_node = TreeNode(keyword, "", "", node_type="or", layer=layer)
            subtree_root_node = tree_node
        else:
            tree_node = TreeNode(keyword, "", "", node_type="or", layer=layer+1)
            subtree_root_node.add_child(tree_node)

        tree_node.set_question(question)
        tree_node.set_answer(raw_answer)

    return subtree_root_node


def generate_aot(dataset_type:str, prior_keywords:list, csv_file:str, num_layers = 3) -> AndOrTree:
    if dataset_type == "movie":
        part_of_speech = "n"
        noun_question_type = "good or bad"

    tree_root = TreeNode(dataset_type, part_of_speech, noun_question_type, node_type="root", layer= 0)
    print("tree root", tree_root.noun_question_type)
    for keyword in prior_keywords:
        #root_child = TreeNode(keyword, part_of_speech, noun_question_type, node_type="or", layer= 1)
        subtree = generate_subtree_from_keyword(keyword, num_layers-1, part_of_speech, noun_question_type, node_type="or", layer=1)
        tree_root.add_child(subtree)

    csv_subtree = generate_subtree_from_csv(csv_file, layer=1)
    tree_root.add_child(csv_subtree)

    return AndOrTree(tree_root)