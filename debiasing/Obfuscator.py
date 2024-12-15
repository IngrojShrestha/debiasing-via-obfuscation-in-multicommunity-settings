import argparse
import os
import shutil
import csv
import numpy as np
import random
import urllib.request
import pandas as pd
from scipy.special import softmax
from nltk.tokenize import TweetTokenizer
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# 007-classifiers
from NULI import NULI
from MIDAS import MIDAS
import mlp_main

# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
class OBF_SentimentClassifier:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.isdir("/path/to/cardiffnlp/"):
            print("Removing existing loaded model", flush=True)
            shutil.rmtree('/path/to/cardiffnlp/')

        task = 'sentiment'
        MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # download label mapping
        self.labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        self.labels = [row[1] for row in csvreader if len(row) > 1]

        # Pre-trained model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        model.save_pretrained(MODEL)

        # run using GPU
        # if torch.cuda.is_available():
        #     print("Using cuda")
        #     self.model = model.cuda()
        # else:
        #     print("Using cpu")
        #     self.model = model

        self.model = model

    # Preprocess text (username and link placeholders)
    def preprocess(self, text):
        new_text = []

        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def get_label_scores(self, text):
        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]

        # example: {positive:0.2, negative:0.5, neutral:0.3}
        preds_dict = {}

        for i in range(scores.shape[0]):
            l = self.labels[ranking[i]]
            s = scores[ranking[i]]
            preds_dict[l] = s
        return preds_dict

    def predict(self, query):

        self.model.eval()

        preds_dict = self.get_label_scores(query)

        # return the prediction label and its corresponding score
        pred_label = None
        pred_score = -1
        for label, score in preds_dict.items():
            if score > pred_score:
                pred_label = label
                pred_score = score

        return pred_label, pred_score

    def predictMultiple(self, test_queries, original_label):

        self.model.eval() # eval mode

        predictions = []
        probs = []

        for test in test_queries:
            preds_dict = self.get_label_scores(test)
            predictions.append(max(preds_dict, key=preds_dict.get))  # store the label with maximum value (max probability)
            probs.append(preds_dict[original_label])

        return predictions, probs

class OBF_BERT:
    def __init__(self, trained_model):
        self.model = trained_model # use GPU as the model is saved to 'CUDA'device
        
    def predict(self, tweet):

        prediction, logit = self.model.predict([tweet]) # text must be provided as a list for simpletransformers
        logit_tensor = torch.tensor(logit).to('cuda')  # Ensure tensor is on the same device as the model
        softmax_output = torch.nn.functional.softmax(logit_tensor, dim=1)
        p_hate = softmax_output[0][1]  # Remain on GPU
        return int(prediction[0]), p_hate.item()  # Convert to item when necessary
    
    def predictMultiple(self, test_queries):
        # Get predictions and logits in one batch operation
        preds, logits = self.model.predict(test_queries)

        # Convert all logits to a PyTorch tensor; assuming logits is a list of lists
        logit_tensor = torch.tensor(logits)

        # the model might be on CUDA, ensure the tensor is also on CUDA
        logit_tensor = logit_tensor.to('cuda')

        # apply softmax on the entire batch tensor along the correct dimension
        softmax_outputs = torch.nn.functional.softmax(logit_tensor, dim=1)

        # extract the probabilities for the 'hate' (toxic) class
        p_hates = softmax_outputs[:, 1]

        # Convert to CPU and numpy array if further non-GPU processing or output is required
        p_hates = p_hates.cpu().numpy()

        predictions = list(preds)

        return predictions, p_hates

class Obfuscator:

    def __init__(self, OO7_classifier='NULI', embedding_file='/path/to/glove.twitter.27B.100d.txt'):

        self.OO7_classifier = OO7_classifier

        if OO7_classifier == 'MIDAS':
            self.obfuscator = MIDAS(
                                    train_data='/path/to/train.csv', # /path/to/offenseval-training-v1.tsv
                                    trained_cnn_model='MIDAS_CNN.pt', # /path/to/MIDAS_CNN.pt
                                    trained_blstm_model='MIDAS_BLSTM.pt', # /path/to/MIDAS_BLSTM.pt
                                    trained_blstmGru_model='MIDAS_BLSTM-GRU.pt') # /path/to/MIDAS_BLSTM-GRU.py
            
        elif OO7_classifier == 'NULI':
            self.obfuscator = NULI(
                                    train_data='/path/to/train.csv', # /path/to/offenseval-training-v1.tsv
                                    trained_model='NULI.pt', # /path/to/NULI.pt
                                    params_file='NULI_params.json') # /path/to/NULI_params.json
            
        elif OO7_classifier == 'MLP':
            self.obfuscator = mlp_main.OBF_MLP(
                                            trained_mlp_model=mlp_main.MODEL_NAME)  # /path/to_mlp_original.pt
            
        elif OO7_classifier == 'BERT_base':
            trained_model='/path/to/bert_main_dwmw17.pt' # path to finetuned BERT model on specific dataset (e.g., DWMW17)
            model = torch.load(trained_model)
            self.obfuscator = OBF_BERT(model)
            
        elif OO7_classifier == 'SentimentClassifier':
            self.obfuscator = OBF_SentimentClassifier()

        # load twitter embeddings
        self.embeddingDict = {}
        embeddings = open(embedding_file, 'r')

        for embedding in embeddings:
            embedding = embedding.strip().split()
            self.embeddingDict[embedding[0]] = [float(x) for x in embedding[1:]]

        embeddings.close()

    def preProcessText(self, text):
        text = text.lower().strip()
        tknzr = TweetTokenizer()
        text = ' '.join(tknzr.tokenize(text))

        return text

    # determines word to be chosen via greedy selection (checking probability changes) and perform random replacement replaces (random replacement)
    def GS_RR(self, text_id, query, original_label):
        
        query = self.preProcessText(query)
    
        # Get initial probability for query
        _, initial_prob = self.obfuscator.predict(query)

        split_query = query.split()
        
        # Generate variations by removing one word at a time
        variations = [' '.join(split_query[:i] + split_query[i + 1:]) for i in range(len(split_query))]
        
        # Get probabilities for all variations
        if self.OO7_classifier =='SentimentClassifier':
            orig_pred, var_probs = self.obfuscator.predictMultiple(variations, original_label)
        else:
            orig_pred, var_probs = self.obfuscator.predictMultiple(variations)
        
        # Calculate the drop in probability after removing each word
        prob_diffs = [initial_prob - cur_prob for cur_prob in var_probs]
        
        # Find the word position with the largest confidence drop
        replace_pos = max(range(len(prob_diffs)), key=lambda i: prob_diffs[i])
        
        # get a random word from vocab to replace word
        rand_pos = random.randint(0, len(self.embeddingDict))
        replace_word = list(self.embeddingDict.keys())[rand_pos]
        replaced = [rand_pos]

        obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos + 1:])
        
        new_prob = None
        new_pred = None
        attempts = 0
        
        # if the query without the word is not original label randomly choose until you find a non-original (non-offenive) replacement
        if (orig_pred[replace_pos] != original_label):
            # keep randomly replacing while prediction is not original label
            new_pred, new_prob = self.obfuscator.predict(obf_query)
            
            attempts += 1 
            while (new_pred == original_label):
                # if all embeddings attempted, break out
                if (len(replaced) == len(self.embeddingDict)):
                    break
                    
                rand_pos = random.choice([x for x in range(0, len(self.embeddingDict)) if x not in replaced])
                replaced.append(rand_pos)
                replace_word = list(self.embeddingDict.keys())[rand_pos]

                obf_query = ' '.join(split_query[:replace_pos] + [replace_word] + split_query[replace_pos + 1:])
                new_pred, new_prob = self.obfuscator.predict(obf_query)
                
                attempts +=1
            
        print('tweet_id:', text_id,
            'original_text_prob:',initial_prob,
            'original_text_label:', original_label,
            'original_label_prob_for_variation:', var_probs[replace_pos], # likelihood after removing the most important word
            'original_pred_for_variation:', orig_pred[replace_pos], # label after removing the most important word
            'max_diff_prob:', max(prob_diffs),
            'word_replaced:',split_query[replace_pos], 
            'replacement:', replace_word,
            'new_prob:', new_prob,
            "new_text_label:", new_pred, 
            "attempts:", attempts, flush=True)

        return obf_query

    def obfuscate(self, text_id, query, original_label):
        return self.GS_RR(text_id, query, original_label)


def get_parser_():
    "separate out parser definition in its own function"
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("STATE", help="provide STATE (e.g., ORIGINAL): ")
    parser_.add_argument("DATA", help="provide DATA (e.g., DWMW17): ")
    parser_.add_argument("RUN", help="provide RUN (e.g., 1,10): ")
    return parser_


def load_model_template(args_):
    import mlp_main as m
    parser = m.get_parser()
    args = parser.parse_args([args_.STATE, args_.DATA, args_.RUN])
    m.setup_args_variables(args)


def main(conversion_file, obfuscator_name, embedding_loc='/path/to/glove.twitter.27B.100d.txt'):
    '''
    saves the obfuscated version of false positive instances

    :param conversion_file: file containing texts to be obfuscated. each line corresponds to false positive instances
    :param obfuscator_name: 007-classifier
    :param embedding_loc: Glove embedding vocabulary
    :return: None.
    '''
    all_tweets = {}
    out_file = open("obf_out.csv", 'w')
    outCSV = csv.writer(out_file, delimiter='\t')

    # load in text to be obfuscated
    with open(conversion_file, 'r') as csvfile:
        tweetreader = csv.reader(csvfile, delimiter='\t')
        for tweet in tweetreader:
            all_tweets[tweet[0]] = tweet

    OO7_classifier = Obfuscator(OO7_classifier=obfuscator_name, embedding_file=embedding_loc)

    # for OBF_SC, we first need to predict original sentiment label and then flip the decision (obfuscation)
    if obfuscator_name == 'SentimentClassifier':
        # update the prediction labels (sentiments)
        for tweet in all_tweets:
            pred_label, _ = OO7_classifier.obfuscator.predict(all_tweets[tweet][1])
            all_tweets[tweet][2] = pred_label # predictions from sentiment classifier

    for tweet in all_tweets:
        # store the obfuscated version of false positive instance
        all_tweets[tweet][1] = OO7_classifier.obfuscate(all_tweets[tweet][0], all_tweets[tweet][1], all_tweets[tweet][2])

        # non-toxic (in our case, greedy-select and random-replace is successful, with no degenerate cases of failing to trick the 007-classifier)
        all_tweets[tweet][2] = 0
        
        out_tweet = all_tweets[tweet]
        outCSV.writerow(out_tweet)
        out_file.flush()
        os.fsync(out_file)
    out_file.close()
