import numpy as np
import torch
from datasets import load_dataset
from transformers import Adafactor
from tqdm.auto import tqdm

from thirdparty.other_models import AnotherNet, RandomDataSampler

batch_size = 16
use_cuda = True and torch.cuda.is_available()
epochs = 100
learning_rate = 2e-5
max_input_length = 100
max_train_samples = 1024
max_test_samples = 1024

def calculate_accuracy(logits, labels):
    return (torch.sum(torch.argmax(logits,dim=1) == labels) / len(labels)).item()

def train(dataset_name:str, model_name:str):
    '''
    Train a model (model_name) on a dataset (dataset_name)
    '''

    # load data
    train_dataset = load_dataset(dataset_name, split='train')
    test_dataset = load_dataset(dataset_name, split='test')

    train_data_sampler = RandomDataSampler(max_train_samples, len(train_dataset)) if max_train_samples is not None and max_train_samples < len(train_dataset) else None
    test_data_sampler = RandomDataSampler(max_test_samples, len(test_dataset)) if max_test_samples is not None and max_test_samples < len(test_dataset) else None
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_data_sampler, batch_size=batch_size)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_data_sampler, batch_size=batch_size)

    #train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    #test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


    # init model
    num_labels = len(set([item["label"] for item in train_dataset]))
    assert num_labels > 1

    model = AnotherNet(model_name, num_labels)
    if use_cuda:
        model = model.cuda()

    loss_fct =  torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        #------------------------------------------Train---------------------------------------------
        model.train()
        train_acc_list = []
        for batch_data in tqdm(train_data_loader):
            batch_x = model.tokenizer(batch_data['text'], padding=True, truncation=True, max_length=max_input_length, return_tensors="pt").input_ids
            batch_label = batch_data['label']

            if use_cuda:
                batch_x = batch_x.to(torch.device("cuda"))
                batch_label = batch_label.to(torch.device("cuda"))

            output_logits = model(batch_x)
            loss = loss_fct(output_logits, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #train_acc_list.append(loss.item())

            #if len(train_acc_list) % 20 == 0:
            #    print("training loss:", loss.item())

            train_acc_list.append(calculate_accuracy(output_logits, batch_label))

        #------------------------------------------Test---------------------------------------------
        model.eval()
        test_acc_list = []
        for batch_data in tqdm(test_data_loader):
            batch_x = model.tokenizer(batch_data['text'], padding=True, truncation=True, max_length=max_input_length, return_tensors="pt").input_ids
            batch_label = batch_data['label']

            if use_cuda:
                batch_x = batch_x.to(torch.device("cuda"))
                batch_label = batch_label.to(torch.device("cuda"))

            output_logits = model(batch_x)
            loss = loss_fct(output_logits, batch_label)

            test_acc_list.append(calculate_accuracy(output_logits, batch_label))

        print("epoch {} train loss {} test acc {}".format(epoch, np.mean(train_acc_list), np.mean(test_acc_list)))


if __name__ == '__main__':
    dataset_name = "rotten_tomatoes" # yelp_polarity rotten_tomatoes
    model_name = "XLNet" # "XLNet" # Roberta

    train(dataset_name, model_name)


    #     loss = None
    # if labels is not None:
    #     if self.num_labels == 1:
    #         #  We are doing regression
    #         loss_fct = MSELoss()
    #         loss = loss_fct(logits.view(-1), labels.view(-1))
    #     else:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))



