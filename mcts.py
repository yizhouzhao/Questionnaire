# monte carlo tree search
import torch

from design.aot import *
from questions.prior_questions import *
from design.algorithm import *
from design.util import dataloader2samples, combine_survey

from thirdparty.other_models import RandomDataSampler

import xgboost as xgb
from sklearn.metrics import precision_score

batch_size = 16

max_train_samples = 128
max_test_samples = None

def run_classifier(X_train, y_train, X_test, y_test, method="xgboost"):
    # set xgboost params
    param = {
        'max_depth': 15,  # the maximum depth of each tree
        'eta': 0.3,  # the training step for each iteration
        'silent': 1,  # logging mode - quiet
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'num_class': 2}  # the number of classes that exist in this datset
    num_round = 20  # the number of training iterations

    # use DMatrix for xgbosot
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    #------------- numpy array ------------------
    # training and testing - numpy matrices
    bst = xgb.train(param, dtrain, num_round)

    # extracting most confident predictions
    preds = bst.predict(dtrain)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    train_score = precision_score(y_train, best_preds, average='macro')
    print("Training Numpy array precision:", train_score)

    preds = bst.predict(dtest)
    # extracting most confident predictions
    best_preds = np.asarray([np.argmax(line) for line in preds])
    test_score = precision_score(y_test, best_preds, average='macro')
    print("Test Numpy array precision:", test_score)

    return train_score, test_score


def train():
    # load aot
    aot = generate_aot("movie", g_movie_piror_keywords[:4], "questions/basicquestions.csv")
    aot_machine = AOTMachine(aot, batch_size=batch_size)
    keyword2survey = {}

    # load data
    dataset_name = "rotten_tomatoes"

    train_dataset = load_dataset(dataset_name, split='train')
    test_dataset = load_dataset(dataset_name, split='test')

    train_data_sampler = RandomDataSampler(max_train_samples, len(train_dataset)) if max_train_samples is not None and max_train_samples < len(train_dataset) else None
    test_data_sampler = RandomDataSampler(max_test_samples,len(test_dataset)) if max_test_samples is not None and max_test_samples < len(test_dataset) else None

    train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_data_sampler, batch_size=batch_size)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_data_sampler, batch_size=batch_size)

    train_samples = dataloader2samples(train_data_loader) # {"text":, "label":}
    test_samples = dataloader2samples(test_data_loader)# {"text":, "label":}

    train_labels = [item["label"] for item in train_samples] #[0 1 2 1 ...]
    test_labels = [item["label"] for item in test_samples]
    

    num_labels = len(set([item["label"] for item in train_dataset]))

    assert num_labels > 1

    # MCTS main
    
    num_iteration = 1000
    # for i in range(iteration):
    keyword2results = {}
    node = aot_machine.aot.sample()
    if node is not None:
        train_survey = aot_machine.conduct_survey_along_samples_and_node(train_samples, node)
        test_survey = aot_machine.conduct_survey_along_samples_and_node(test_samples, node)

        keyword2results[node.keyword] = {"train": train_survey, "test": test_survey}

        # train
        train_array, test_array = combine_survey(keyword2results)
        train_score, test_score = run_classifier(train_array, train_labels, test_array, test_labels)
        
         


        




