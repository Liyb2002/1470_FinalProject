import os
import numpy as np
import pandas as pd
from simplet5 import SimpleT5
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from random import sample


trainning_df = pd.read_csv(
    '../newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv')
MAX_LEN = 512
SUMMARY_LEN = 150
TRAINNING_SIZE = 5000
trainning_df = trainning_df.iloc[0:TRAINNING_SIZE,:].copy()
trainning_article_ls = list(trainning_df['article'])
trainning_highlight_ls = list(trainning_df['highlights'])
del trainning_df



df = pd.DataFrame(columns=['target_text', 'source_text'])
df['target_text'] = trainning_highlight_ls
df['source_text'] = ['summarize: '+item for item in trainning_article_ls]
model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")

MAX_EPOCHS = 5
model.train(train_df=df[0:(int)(0.7*TRAINNING_SIZE)],
            eval_df=df[(int)(0.7*TRAINNING_SIZE):TRAINNING_SIZE],
            source_max_token_len=MAX_LEN,
            target_max_token_len=SUMMARY_LEN,
            batch_size=8, max_epochs=MAX_EPOCHS, use_gpu=True)

model_path = ''
rootdir = 'outputs/'
for it in os.scandir(rootdir):
    if it.is_dir():
        if 'simplet5-epoch-'+(str)(MAX_EPOCHS-1) in it.path:
            model_path = it.path
            print(model_path)

model.load_model("t5", "./"+model_path, use_gpu=True)
no_tune_model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')


test_df = pd.read_csv(
    '../newspaper-text-summarization-cnn-dailymail/cnn_dailymail/test.csv')
MAX_LEN = 512
SUMMARY_LEN = 150
TEST_SIZE = 500
test_df = trainning_df.iloc[0:TEST_SIZE, :].copy()
test_article_ls = list(test_df['article'])
test_highlight_ls = list(test_df['highlights'])
del test_df


df = pd.DataFrame(columns=['target_text', 'source_text'])
df['target_text'] = test_highlight_ls
df['source_text'] = ['summarize: '+item for item in test_article_ls]

device = torch.device('gpu')
for index in sample(list(np.arange(len(trainning_article_ls))), 5):
    print('Original Text : ')
    print(trainning_article_ls[index])

    print('\n\nSummary Text : ')
    print(trainning_highlight_ls[index])

    print('\n\nFine tuned Predicted Summary Text : ')
    print(model.predict(trainning_article_ls[index]))

    print('\n\nNot Fine tuned Predicted Summary Text : ')
    preprocess_text = trainning_article_ls[index].strip().replace("\n", "")
    t5_prepared_Text = "summarize: "+preprocess_text
    tokenized_text = tokenizer.encode(
        t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = no_tune_model.generate(tokenized_text,
                                         num_beams=4,
                                         no_repeat_ngram_size=2,
                                         min_length=30,
                                         max_length=SUMMARY_LEN,
                                         early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
