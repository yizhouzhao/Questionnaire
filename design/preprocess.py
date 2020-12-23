# preprocess dataset
from datasets import Dataset
from tqdm.auto import tqdm

def AG_NEWS(dataset:Dataset):
    '''
    Pre-processing ag_news dataset
    '''
    for item in tqdm(dataset):
        item['text'] = item['text'].replace("\\", " ").replace(" -- ",": ").replace(" - ",": ").replace("  "," ").lower()
