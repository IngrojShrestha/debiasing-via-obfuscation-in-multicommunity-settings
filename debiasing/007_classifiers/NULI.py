# https://github.com/jonrusert/robustnessofoffensiveclassifiers
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch_transformers import *
import time
from torch.utils.data import Dataset
import copy
from torch.optim import lr_scheduler
import os
import csv
import datetime
import json

log = 'NULIlog'

# Example from https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
class BertForSequenceClassification(nn.Module):

    def __init__(self, config, num_labels=2):
        super(BertForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

def load_dataset(trFile=None, teFile=None):
    labelsAsNums = {'HATE': 1, 'NOT': 0}
    numsAsLabels = {"1": 'HATE', '0': 'NOT'}
    labelNum = 2
    numTweets = 0
    testTweets = []

    x_train = []
    y_train = []
    x_test = []

    # NULI used a max_sequence_length of 64
    max_sequence_length = 64

    # load in train tweets and corresponding labels
    if (trFile):
        with open(trFile, 'r') as csvfile:
            tweetreader = csv.reader(csvfile, delimiter='\t')
            for tweet in tweetreader:
                text = tweet[1].lower().strip()
                x_train.append(text)
                y_train.append(labelsAsNums[tweet[2]])

    # load in test tweets and corresponding labels
    if (teFile):
        with open(teFile, 'r') as csvfile:
            tweetreader = csv.reader(csvfile, delimiter='\t')
            for tweet in tweetreader:
                text = tweet[1].lower().strip()
                testTweets.append(tweet)
                x_test.append(text)

    return x_train, y_train, x_test, labelNum, testTweets, labelsAsNums, numsAsLabels, max_sequence_length

class NULI:

    def __init__(self, train_data='train.tsv', trained_model='NULI.pt',
                 params_file='NULI_params.json'):

        # load in params
        params_in = open(params_file)
        params_lines = params_in.readlines()
        params = json.loads(params_lines[0])

        self.labelNum = params['labelNum']
        self.labelsAsNums = params['labelsAsNums']
        self.numsAsLabels = params['numsAsLabels']
        self.max_seq_length = params['max_seq_length']

        # Load pre-trained tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load pre-trained model
        self.model = torch.load(trained_model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    # preprocess test_query
    def preprocessQuery(self, test_query):
        text = test_query.lower().strip()
        return text

    # convert pre-processed test query into tokenized review
    def tokenizeQuery(self, text):
        tokenized_review = self.tokenizer.tokenize(text)

        if len(tokenized_review) > self.max_seq_length:
            tokenized_review = tokenized_review[:self.max_seq_length]

        ids_review = self.tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (self.max_seq_length - len(ids_review))

        ids_review += padding

        assert len(ids_review) == self.max_seq_length

        return ids_review

    def predict(self, test_query):

        text = self.preprocessQuery(test_query)
        ids_review = self.tokenizeQuery(text)

        test = []
        test.append(ids_review)

        self.model.eval() # eval mode
        
        # make prediction
        predicted = []
        probs = []

        tests = torch.tensor(test).to('cuda')
        outputs = self.model(tests)
        outputs = outputs.detach().cpu().numpy().tolist()

        for cur_output in outputs:
            predicted.append(cur_output.index(max(cur_output)))
            probs.append(cur_output[self.labelsAsNums['HATE']])

        # should only be length 1 since only one query was sent in
        predicted_label = self.numsAsLabels[str(predicted[0])]
        label_prob = probs[0]

        return predicted_label, label_prob

    # prediction on multiple queries at once
    def predictMultiple(self, test_queries):

        test = []
        for test_query in test_queries:
            text = self.preprocessQuery(test_query)
            ids_review = self.tokenizeQuery(text)
            test.append(ids_review)

        self.model.eval()

        # make prediction
        predicted = []
        probs = []

        tests = torch.tensor(test).to('cuda')
        outputs = self.model(tests)
        outputs = outputs.detach().cpu().numpy().tolist()

        for cur_output in outputs:
            predicted.append(self.numsAsLabels[str(cur_output.index(max(cur_output)))])
            probs.append(cur_output[self.labelsAsNums['HATE']])

        return predicted, probs


class text_dataset(Dataset):

    def __init__(self, x_y_list, max_seq_length, transform=None):
        self.x_y_list = x_y_list
        self.max_seq_length = max_seq_length
        self.transform = transform

    def __getitem__(self, index):
        max_seq_length = self.max_seq_length
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])

        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]

        ids_review = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))

        ids_review += padding

        assert len(ids_review) == max_seq_length

        ids_review = torch.tensor(ids_review)

        sentiment = self.x_y_list[1][index] 
        list_of_labels = [torch.from_numpy(np.array(sentiment))]

        return ids_review, list_of_labels[0]

    def __len__(self):
        return len(self.x_y_list[0])


def train_model(model, criterion, optimizer, scheduler, dataloaders_dict, device, num_epochs=2):
    print('Training model...')

    if torch.cuda.is_available():
        model = model.cuda()

    for epoch in range(num_epochs):
        nlog = open(log, 'a')
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        nlog.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        nlog.write(str(datetime.datetime.now()) + '\n')
        nlog.write('-' * 10 + '\n')
        nlog.close()

        scheduler.step()
        model.train()  # set train mode

        phase = 'train'
        
        # Iterate over data
        for inputs, labels in dataloaders_dict[phase]:
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad() # reset gradients to zero before backpropagation

            outputs = model(inputs) # forward pass
            loss = criterion(outputs, labels)
            loss.backward() # compute gradients via backpropagation
            optimizer.step() # update model parameters using the computed gradients

    print("Finished training")
    return model


def main(trainFile, testFile, train_test, out_file=None):
    nlog = open(log, 'a')
    nlog.write('loading data...\n')
    nlog.write(str(datetime.datetime.now()) + '\n')
    nlog.flush()
    os.fsync(nlog)
    nlog.close()

    if (train_test == 'train' or train_test == 'both'):
        x_train, y_train, x_test, labelNum, testTweets, labelsAsNums, numsAsLabels, max_seq_length = load_dataset(trainFile, testFile)
    else:
        # load in params
        params_in = open('NULI_params.json')
        params_lines = params_in.readlines()
        params = json.loads(params_lines[0])

        labelNum = params['labelNum']
        labelsAsNums = params['labelsAsNums']
        numsAsLabels = params['numsAsLabels']
        max_seq_length = params['max_seq_length']
        _, _, x_test, _, testTweets, _, _, _ = load_dataset(None, testFile)

    nlog = open(log, 'a')
    nlog.write('finished loading data...\n')
    nlog.write(str(datetime.datetime.now()) + '\n')
    nlog.flush()
    os.fsync(nlog)
    nlog.close()
    batch_size = 4
    
    # Load pre-trained model tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (train_test == 'train' or train_test == 'both'):
        train_lists = [x_train, y_train]

        training_dataset = text_dataset(x_y_list=train_lists, max_seq_length=max_seq_length)

        dataloaders_dict = {
            'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)}

        num_labels = labelNum + 1
        config = BertConfig()  
        model = BertForSequenceClassification(config, num_labels)
        lrlast = .001
        lrmain = .00001
        optim1 = optim.Adam(
            [
                {"params": model.bert.parameters(), "lr": lrmain},
                {"params": model.classifier.parameters(), "lr": lrlast},

            ])

        # all parameters are being optimized
        optimizer_ft = optim1
        criterion = nn.CrossEntropyLoss()

        # Decay the learning rate by a factor of 0.1 every 3 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)

        model_ft1 = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders_dict, device,
                                num_epochs=3)

        torch.save(model_ft1, 'NULI.pt')

        # create a json to store dictionaries and information for loading use
        out_info = {'labelNum': labelNum, 'labelsAsNums': labelsAsNums, 'numsAsLabels': numsAsLabels,
                    'max_seq_length': max_seq_length}

        outtmp = json.dumps(out_info)

        outfile = open('NULI_params.json', 'w')
        outfile.write(outtmp)
        outfile.flush()
        os.fsync(outfile)
        outfile.close()

    if (train_test == 'test' or train_test == 'both'):

        model_ft1 = torch.load('NULI.pt')

        if torch.cuda.is_available():
            model_ft1 = model_ft1.cuda()

        model_ft1.eval()

        # load test set for predictions
        tests = []
        for cur_x in x_test:

            tokenized_review = tokenizer.tokenize(cur_x)

            if len(tokenized_review) > max_seq_length:
                tokenized_review = tokenized_review[:max_seq_length]

            ids_review = tokenizer.convert_tokens_to_ids(tokenized_review)
            padding = [0] * (max_seq_length - len(ids_review))
            ids_review += padding

            assert len(ids_review) == max_seq_length

            tests.append(ids_review)

        if (out_file == None):
            prediction_file = open('NULI_predictionsOut', 'w')
        else:
            prediction_file = open(out_file, 'w')

        predicted = []
        tests = torch.tensor(tests)

        dataloaders_dict = {
            'test': torch.utils.data.DataLoader(tests, batch_size=batch_size, shuffle=False, num_workers=0)}

        outputs = []
        for inputs, labels in dataloaders_dict['test']:
            inputs = inputs.to(device)
            out = model_ft1(inputs)
            out = out.detach().cpu().numpy().tolist()
            outputs.append(out)

        for cur_output in outputs:
            predicted.append(cur_output.index(max(cur_output)))

        # write predictions to file
        for j in range(len(predicted)):
            pred = numsAsLabels[str(predicted[j])]

            prediction_file.write(str(testTweets[j][0]) + ',' + pred + '\n')

        prediction_file.close()
