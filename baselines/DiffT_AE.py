import os
import sys
import time
import random
import math
import numpy as np
import pandas as pd
from collections import Counter
import bcolz
import pickle
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch import autograd
from nltk import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    print("enable seed")
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################################
# model parameters
MAX_SEQ_LEN = None
VOCAB_SIZE = None
NUM_CLASSES = None
VOCAB = None

LEARNING_RATE = 1e-4
enable_l1 = True
enable_l2 = True
NUM_EPOCHS = 50
HIDDEN_SIZE = 128
BATCH_SIZE = 64
EMBEDDING_DIM = 300
DROPOUT = 0.50
padding = 'post'
truncating = 'post'
ACTIVATION = nn.Sigmoid()
is_early_stopping = True
patience = 5
is_padding_idx = False  # True prevent padding index (0) to be not trainable in Embedding()
UNK_embedding_lst = ['zeros', 'random']
UNK_embedding = UNK_embedding_lst[1]

MAX_SEQ_LEN_types = ['max', 'avg', 'min_max']
MAX_SEQ_LEN_type = MAX_SEQ_LEN_types[2]

is_grad_clip = True

if enable_l1:
    l1_alpha = 1e-6
else:
    l1_alpha = None

if enable_l2:
    l2_alpha = 1e-5
else:
    l2_alpha = None

GLOVE_PATH = "/path/to/glove/" # this folder contains 6B.300.dat, 6B.300_words.pkl, 6B.300_idx.pkl

porter_stemmer = PorterStemmer()

##################################################################################
# DiffT settings
# wp_lst = [0.1, 0.32, 1, 3.2, 10]
B_old = None  # bias on validation set to track the best model
N = 1000
RUN_NUM = 1
df_train_last_best = None
num_duplicated = 0
num_removed = 0
##################################################################################
# user input
DATA = sys.argv[1]
wp = float(sys.argv[2])
##################################################################################
# INPUT and OUTPUT files
BASE_OUT_PATH = f"../output/DiffT/debias_data_prediction/wp_{wp}/"

if not os.path.exists(BASE_OUT_PATH):
    os.makedirs(BASE_OUT_PATH)

MODEL_NAME = BASE_OUT_PATH + f"mlp_{DATA.lower()}_{wp}.pt"

PREDICTION_TRAIN_OUT = BASE_OUT_PATH + f"train_predictions_{DATA.lower()}.csv"
PREDICTION_TRAIN_PROB_OUT = BASE_OUT_PATH + f"train_predictions_{DATA.lower()}.csv"

PREDICTION_VALID_OUT = BASE_OUT_PATH + f"predictions_valid_{DATA.lower()}.csv"
PREDICTION_VALID_PROB_OUT = BASE_OUT_PATH + f"predictions_valid_prob_{DATA.lower()}.csv"

PREDICTION_TEST_OUT = BASE_OUT_PATH + f"predictions_{DATA.lower()}.csv"
PREDICTION_TEST_PROB_OUT = BASE_OUT_PATH + f"predictions_prob_{DATA.lower()}.csv"

NEW_TRAIN_FILE = BASE_OUT_PATH + f'{DATA.lower()}_debiased_train.csv'

TRAIN_FILE_PATH = f"../data/{DATA}/{DATA.lower()}_train.csv"
VAL_FILE_PATH = f"../data/{DATA}/{DATA.lower()}_valid.csv"
TEST_FILE_PATH = f"../data/{DATA}/{DATA.lower()}_test.csv"

MAIN_FILE = f"../data/{DATA}/{DATA.lower()}_tweet_label_group_binary.csv"

CHECKPOINT_PATH = BASE_OUT_PATH + f"checkpoint_{DATA.lower()}.pt"

##################################################################################
class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


# Defining MLP architecture
class MLP(nn.Module):  # inheriting from nn.Module!
    def __init__(self):
        super().__init__()
        
        # load pretrained glove embeddings and create embedding weight matrix
        glove = load_pretrained_glove()
        embedding_matrix = glove_weight_matrix(glove)

        if is_padding_idx:
            self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=0)
        else:
            self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = True # Allow fine-tuning of embeddings

        # Linear function 1
        self.fc1 = nn.Linear(EMBEDDING_DIM * MAX_SEQ_LEN, HIDDEN_SIZE)

        # Non-linearity
        self.activation = ACTIVATION

        self.dropout = nn.Dropout(p=DROPOUT)

        # Linear function 2
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, text):
        
        # Embed the input text using the embedding layer
        embedded = self.embedding(text)
        
        x = embedded.view(embedded.shape[0], -1)

        # Linear function 1
        x = self.fc1(x)

        # Non-linearity 1
        embedRepr = self.activation(x)

        # Dropout
        x = self.dropout(embedRepr)

        # Linear function 2
        preds = self.fc2(x)

        return preds

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def load_data():
    print("Loading data \n")
    df_train = pd.read_csv(TRAIN_FILE_PATH, sep="\t")
    df_test = pd.read_csv(TEST_FILE_PATH, sep="\t")
    df_valid = pd.read_csv(VAL_FILE_PATH, sep="\t")

    df_train['tweet_id'] = df_train['tweet_id'].astype(str)
    df_test['tweet_id'] = df_test['tweet_id'].astype(str)
    df_valid['tweet_id'] = df_valid['tweet_id'].astype(str)

    # convert tweet to lowercase
    df_train["tweet"] = df_train["tweet"].apply(lambda x: x.lower())
    df_test["tweet"] = df_test["tweet"].apply(lambda x: x.lower())
    df_valid["tweet"] = df_valid["tweet"].apply(lambda x: x.lower())

    # stemming
    df_train['tweet'] = df_train['tweet'].apply(stem_sentences)
    df_test['tweet'] = df_test['tweet'].apply(stem_sentences)
    df_valid['tweet'] = df_valid['tweet'].apply(stem_sentences)

    return df_train, df_test, df_valid


def load_pretrained_glove():
    vectors = bcolz.open(f'{GLOVE_PATH}/6B.300.dat')[:]
    words = pickle.load(open(f'{GLOVE_PATH}/6B.300_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{GLOVE_PATH}/6B.300_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}
    return glove

def glove_weight_matrix(glove):
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

    for word, index in VOCAB.items():
        try:
            embedding_matrix[index] = glove[word]
        except KeyError:
            if UNK_embedding == 'zeros':
                embedding_matrix[index] = np.zeros((EMBEDDING_DIM,))  # sets zeros for unknown and padded index
            elif UNK_embedding == 'random':
                embedding_matrix[index] = np.random.normal(scale=0.01, size=(EMBEDDING_DIM,))

    return embedding_matrix

def create_tokenizer(lines):
    tokenizer = Tokenizer(oov_token=True)
    tokenizer.fit_on_texts(lines)
    return tokenizer

def sample_train(df_old_train, sampling_dict, N):
    global num_duplicated, num_removed
    '''
    Returns new training set after sampling
    '''
    num_duplicated = 0
    num_removed = 0

    df_new_train = pd.DataFrame()

    for index, (tweet_id, sampling) in enumerate(sampling_dict.items()):
        if index > N - 1:
            df_new_train = df_new_train.append(df_old_train[df_old_train['tweet_id'] == tweet_id])
        else:
            if sampling == 1:  # duplicate
                df_new_train = df_new_train.append(df_old_train[df_old_train['tweet_id'] == tweet_id])
                df_new_train = df_new_train.append(df_old_train[df_old_train['tweet_id'] == tweet_id])

                # change tweet_id datatype from int to str in order to save string tweet_id
                # e.g. 123n where n represents duplicate and of tweet_id 123
                df_new_train = df_new_train.astype({"tweet_id": str})

                # extract last row
                row = df_new_train.tail(1)
                temp_tweet_id = row['tweet_id'].values[0]

                # change tweet_id
                df_new_train.reset_index(inplace=True, drop=True)
                df_new_train.at[df_new_train.index[-1], 'tweet_id'] = str(temp_tweet_id) + "n"

                num_duplicated += 1

            elif sampling == -1:  # drop
                num_removed += 1
                continue

    df_new_train = df_new_train.reset_index(drop=True)
    return df_new_train


def get_test_performance(TEST_DATA_PATH, PREDS_PATH, dataset):
    print(f"\nGet bias on {dataset}...")

    def load_gs_preds():
        print("load_gs_preds\n")
        with open(PREDS_PATH, 'rb') as f:
            preds = pickle.load(f)

        df = pd.read_csv(TEST_DATA_PATH, sep='\t')
        y_true = df['label'].values.tolist()

        y_pred = []
        for ps in preds:
            y_pred.append(ps)

        return y_true, y_pred

    def load_test_data():
        print("load_test_data\n")
        df = pd.read_csv(TEST_DATA_PATH, sep="\t")
        return df

    def compute_eval_metrics(tn, fp, fn, tp):
        TPR = round(tp / (tp + fn), 3)
        FPR = round(fp / (fp + tn), 3)
        FNR = round(fn / (fn + tp), 3)
        PPV = round(tp / (tp + fp), 3)
        NPV = round(tn / (tn + fn), 3)

        print("TPR: ", TPR)
        print("FPR: ", FPR)
        print("FNR: ", FNR)
        print("PPV: ", PPV)
        print("NPV: ", NPV)
        return TPR, FPR, FNR, PPV, NPV

    def print_confusion_matrix_group(df_all, group_label):
        '''
        group = {0:'African American', 1:'Hispanic', 2:'Asian', 3: 'White'}
        '''
        print()
        print("-" * 50)
        print(group_label)
        print("-" * 50)

        if group_label == 'ae':
            df_group_level = df_all[df_all['group'] == 0]
        elif group_label == 'hispanic':
            df_group_level = df_all[df_all['group'] == 1]
        elif group_label == 'asian':
            df_group_level = df_all[df_all['group'] == 2]
        elif group_label == 'white':
            df_group_level = df_all[df_all['group'] == 3]
        else:
            return "Not valid group name"

        y_true = list(df_group_level['gs'].values)
        y_pred = list(df_group_level['preds'].values)

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = round(accuracy_score(y_true, y_pred), 3)

        print("Accuracy/microF1: ", accuracy)
        print("macro_f1: ", round(macro_f1, 3))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("\nConfusuion matrix: ")
        print("TP: ", tp)
        print("FP: ", fp)
        print("FN: ", fn)
        print("TN: ", tn)

        print("\n Fairness metrics: ")
        TPR, FPR, FNR, PPV, NPV = compute_eval_metrics(tn, fp, fn, tp)

        print(TPR, FPR, FNR, PPV, NPV)

    def test_main():
        print("*" * 10 + "Bias on testing set begins" + "*" * 10)
        y_true, y_pred = load_gs_preds()

        df = load_test_data()

        # extract map tweet_id to corresponding gs, preds and group
        df_all = pd.DataFrame(columns=['tweet_id', 'gs', 'preds', 'group'])
        for index, row in df.iterrows():
            tweet_id = row['tweet_id']
            grp = row['group']
            preds = y_pred[index]
            gs = y_true[index]
            df_all = df_all.append({'tweet_id': tweet_id, 'gs': gs, 'preds': preds, 'group': grp}, ignore_index=True)

        # obtain confusion matrix for overall dataset
        print("Overall confusion matrix (all groups)")
        TPs = df_all[(df_all['gs'] == 1) & (df_all['preds'] == 1)].shape[0]
        TNs = df_all[(df_all['gs'] == 0) & (df_all['preds'] == 0)].shape[0]
        FNs = df_all[(df_all['gs'] == 1) & (df_all['preds'] == 0)].shape[0]
        FPs = df_all[(df_all['gs'] == 0) & (df_all['preds'] == 1)].shape[0]
        print("TP: ", TPs)
        print("FP: ", FPs)
        print("FN: ", FNs)
        print("TN: ", TNs)
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        overall_cf = "\makecell[l]{" + str(TPs) + "," + str(FPs) + ",\\\\" + str(FNs) + "," + str(TNs) + "}"
        print("Overall CF: ", overall_cf)

        print("Accuracy/microF1: ", round(accuracy_score(y_true, y_pred), 3))
        print("macro_f1: ", round(macro_f1, 3))

        print_confusion_matrix_group(df_all, "ae")
        print_confusion_matrix_group(df_all, "white")
        print_confusion_matrix_group(df_all, 'hispanic')
        print_confusion_matrix_group(df_all, 'asian')
        print("*" * 10 + "Bias on testing set ends" + "*" * 10)

    test_main()


def get_val_bias(DATA_PATH, PREDS_PATH, df_main, q2, dataset):
    print(f"\nGet bias on {dataset}...")

    def compute_eval_metrics(tn, fp, fn, tp):
        TPR = round(tp / (tp + fn), 3)
        FPR = round(fp / (fp + tn), 3)
        FNR = round(fn / (fn + tp), 3)
        PPV = round(tp / (tp + fp), 3)
        NPV = round(tn / (tn + fn), 3)
        return TPR, FPR, FNR, PPV, NPV

    def load_data():
        df = pd.read_csv(DATA_PATH, sep="\t")
        df['tweet_id'] = df['tweet_id'].astype(str)
        return df

    def get_FPR(df_all, group_label):
        '''
        group = {0:'African American', 1:'super'}
        '''
        if group_label == 'grp_sub':
            df_group_level = df_all[df_all['group'] == 0]
        elif group_label == 'grp_super':
            df_group_level = df_all[df_all['group'] == 1]
        else:
            return "Not valid group name"

        y_true = list(df_group_level['gs'].values)
        y_pred = list(df_group_level['preds'].values)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        TPR, FPR, FNR, PPV, NPV = compute_eval_metrics(tn, fp, fn, tp)
        return FPR

    def load_gs_preds():
        with open(PREDS_PATH, 'rb') as f:
            preds = pickle.load(f)

        df = pd.read_csv(DATA_PATH, sep='\t')
        y_true = df['label'].values.tolist()

        y_pred = []
        for ps in preds:
            y_pred.append(ps)

        return y_true, y_pred

    def temp_main():
        y_true, y_pred = load_gs_preds()

        df = load_data()

        # extract map tweet_id to corresponding gs, preds and group
        df_all = pd.DataFrame(columns=['tweet_id', 'gs', 'preds', 'group'])
        for index, row in df.iterrows():
            tweet_id = row['tweet_id']
            preds = y_pred[index]
            gs = y_true[index]

            # extract group based on q2
            pAE = df_main[df_main['tweet_id'] == tweet_id]['AE'].values[0]
            if pAE > q2:
                grp = 0  # AE
            else:
                grp = 1  # non-AE(super group)

            df_all = df_all.append({'tweet_id': tweet_id, 'gs': gs, 'preds': preds, 'group': grp}, ignore_index=True)

        FPR_sub = get_FPR(df_all, "grp_sub")
        FPR_super = get_FPR(df_all, "grp_super")

        B = round((FPR_sub - FPR_super), 3)
        return B

    B = temp_main()
    return B


def train_predictions(model, data_loader, df, dataset):
    model.eval()

    with torch.no_grad():

        preds = []
        preds_prob = []

        # predict all samples in the test set batch per batch
        for i, (x_batch, y_batch, index) in enumerate(data_loader):
            y_temp = model(x_batch)
            y_pred = y_temp.detach()

            m = nn.Softmax(dim=1)
            prob = m(y_pred)
            x, predicted_label = torch.max(y_temp.data, 1)

            preds.append(predicted_label.data.cpu().numpy()[0])
            preds_prob.append(prob.cpu().numpy()[0])

    with open(PREDICTION_TRAIN_OUT[:-4] + ".pkl", 'wb') as f:
        pickle.dump(preds, f)

    with open(PREDICTION_TRAIN_PROB_OUT[:-4] + ".pkl", 'wb') as f:
        pickle.dump(preds_prob, f)

    f = open(PREDICTION_TRAIN_OUT, 'w')
    df = df.reset_index(drop=True)

    for index, row in df.iterrows():
        tweet_id = row['tweet_id']
        gs = row['label']
        preds_ = preds[index]
        preds_prob_ = preds_prob[index]

        out = str(tweet_id) + "\t" + str(gs) + "\t" + str(preds_prob_) + "\t" + str(preds_)
        f.write(out + "\n")
    print(f"Predictions on: {dataset} dataset completed!!")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_max_text_length(lengths):
    lengths = sorted(lengths)

    dist = dict(Counter(lengths))

    temp_dist = dict()

    for k, v in dist.items():
        if v >= 5:
            temp_dist[k] = v

    return max(list(temp_dist.keys()))


def create_datasets(df_train, df_test, df_valid):
    global VOCAB_SIZE, NUM_CLASSES, VOCAB, MAX_SEQ_LEN

    print("Creating vocabulary...\n")
    tokenizer = create_tokenizer(df_train['tweet'].values)  # [text1, text2, text3]
    VOCAB = tokenizer.word_index

    VOCAB_SIZE = len(VOCAB) + 1

    NUM_CLASSES = len(df_train['label'].unique())

    lengths = [len(s.split()) for s in df_train['tweet'].values]

    if MAX_SEQ_LEN_type == 'max':
        MAX_SEQ_LEN = max([len(s.split()) for s in df_train['tweet'].values])
    elif MAX_SEQ_LEN_type == 'avg':
        MAX_SEQ_LEN = math.ceil(sum(lengths) / len(lengths))
    elif MAX_SEQ_LEN_type == 'min_max':
        MAX_SEQ_LEN = get_max_text_length(lengths)

    print("Labels: ", df_train['label'].unique())
    print(MAX_SEQ_LEN, VOCAB_SIZE, NUM_CLASSES)

    print("Converting train data to sequence...\n")
    train_sequences = tokenizer.texts_to_sequences(df_train['tweet'].values)  # word2idx
    train_x = pad_sequences(train_sequences, maxlen=MAX_SEQ_LEN, padding=padding, truncating=truncating)  # padding
    train_y = df_train['label'].values

    print("Converting valid data to sequence...\n")
    valid_sequences = tokenizer.texts_to_sequences(df_valid['tweet'].values)  # word2idx
    valid_x = pad_sequences(valid_sequences, maxlen=MAX_SEQ_LEN, padding=padding, truncating=truncating)  # padding
    valid_y = df_valid['label'].values

    print("Converting test data to sequence...\n")
    test_sequences = tokenizer.texts_to_sequences(df_test['tweet'].values)  # word2idx
    test_x = pad_sequences(test_sequences, maxlen=MAX_SEQ_LEN, padding=padding, truncating=truncating)  # padding
    test_y = df_test['label'].values

    x_train = torch.tensor(train_x, dtype=torch.long).to(DEVICE)
    y_train = torch.tensor(train_y, dtype=torch.long).to(DEVICE)

    x_valid = torch.tensor(valid_x, dtype=torch.long).to(DEVICE)
    y_valid = torch.tensor(valid_y, dtype=torch.long).to(DEVICE)

    x_test = torch.tensor(test_x, dtype=torch.long).to(DEVICE)
    y_test = torch.tensor(test_y, dtype=torch.long).to(DEVICE)

    train = MyDataset(torch.utils.data.TensorDataset(x_train, y_train))
    valid = MyDataset(torch.utils.data.TensorDataset(x_valid, y_valid))
    test = MyDataset(torch.utils.data.TensorDataset(x_test, y_test))

    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, shuffle=False)

    train_loader_temp = torch.utils.data.DataLoader(train, shuffle=False)
    valid_loader_temp = torch.utils.data.DataLoader(valid, shuffle=False)

    return train_loader, valid_loader, test_loader, train_loader_temp, valid_loader_temp


def get_current_date_time(status):
    '''
    :param status: start, end
    :return:
    '''
    from datetime import datetime

    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{status} time: ", dt_string)


def l1_penalty(params, l1_lambda=1e-6):
    """Returns the L1 penalty of the params."""
    l1_norm = sum(p.abs().sum() for p in params)
    return l1_lambda * l1_norm


def train_model(model, optimizer, criterion, patience, train_loader, valid_loader):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    get_current_date_time(status="Start")

    # capture training time
    start_time = time.time()

    if is_early_stopping:
        # initialize the early_stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=CHECKPOINT_PATH)

    with autograd.detect_anomaly():

        # for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        for epoch in (range(1, NUM_EPOCHS + 1)):
            # print("Epoch: ", epoch)

            ###################
            # train the model #
            ###################

            model.train()  # prep model for training

            for i, (x_batch, y_batch, index) in enumerate(train_loader):
                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(x_batch)

                # m = nn.Softmax(dim=1)
                # prob = m(outputs)
                # x, predicted_label = torch.max(outputs.data, 1)

                # # Calculate Loss: softmax --> cross entropy loss
                if enable_l1:
                    loss = criterion(outputs, y_batch) + l1_penalty(model.parameters())
                else:
                    loss = criterion(outputs, y_batch)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                if is_grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                # # perform a single optimization step (parameter update)
                optimizer.step()

                # record training loss
                train_losses.append(loss.item())

            ######################
            # validate the model #
            ######################
            model.eval()  # prep model for evaluation
            # valid_preds = np.zeros((x_valid.size(0)))
            for i, (x_batch, y_batch, index) in enumerate(valid_loader):
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = model(x_batch)
                # y_pred = y_temp.detach()

                # calculate the loss
                if enable_l1:
                    loss = criterion(outputs, y_batch) + l1_penalty(model.parameters())
                else:
                    loss = criterion(outputs, y_batch)

                # record validation loss
                valid_losses.append(loss.item())

            # print training/validation statistics
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            elapsed_time = time.time() - start_time

            print('Epoch {}/{} \t train_loss={:.5f} \t val_loss={:.5f} \t time={:.2f}s'.format(
                epoch, NUM_EPOCHS, train_loss, valid_loss, elapsed_time))

            # clear lists to track next epoch
            train_losses = []
            valid_losses = []

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if is_early_stopping:
                early_stopping(valid_loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    if is_early_stopping:
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

    get_current_date_time(status="End")

    print("Time taken to train the model: " + str(time.time() - start_time))

    return model, avg_train_losses, avg_valid_losses


def visualize_loss_early_stopping(train_loss, valid_loss, RUN_NUM):
    y_limit = max(max(valid_loss), max(train_loss))

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, y_limit)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(BASE_OUT_PATH + f'loss_plot_{DATA.lower()}_{RUN_NUM}.png',
                bbox_inches='tight')


def main(df_train, df_test, df_valid, StopFlag, RUN_NUM):
    global N, B_old, df_train_last_best

    set_seed(42)

    print("*" * 50 + f"RUN{RUN_NUM}" + "*" * 50)

    train_loader, valid_loader, test_loader, train_loader_temp, valid_loader_temp = create_datasets(df_train, df_test,
                                                                                                    df_valid)
    # TRAIN
    print("\nTraining model...")
    model = MLP()
    if torch.cuda.is_available():
        model = model.cuda()

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # L2 regularization: weight_decay
    if enable_l2:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=l2_alpha)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = torch.nn.CrossEntropyLoss()

    model, train_loss, valid_loss = train_model(model, optimizer, criterion, patience, train_loader, valid_loader)

    visualize_loss_early_stopping(train_loss, valid_loss, RUN_NUM)

    # save model
    print("Saving trained model")
    torch.save(model, MODEL_NAME)

    model.eval()
    # predict all samples in the test set batch per batch
    test_preds = []
    test_preds_prob = []
    for i, (x_batch, y_batch, index) in enumerate(test_loader):
        # print("========TEST", x_batch.shape)
        y_temp = model(x_batch)
        y_pred = y_temp.detach()

        m = nn.Softmax(dim=1)
        prob = m(y_pred)
        x, predicted_label = torch.max(y_temp.data, 1)

        test_preds.append(predicted_label.data.cpu().numpy()[0])
        test_preds_prob.append(prob.cpu().numpy()[0])

    # predict all samples in the validation set batch per batch
    val_preds = []
    val_preds_prob = []
    for i, (x_batch, y_batch, index) in enumerate(valid_loader_temp):
        # print("========TEST", x_batch.shape)
        y_temp = model(x_batch)
        y_pred = y_temp.detach()

        m = nn.Softmax(dim=1)
        prob = m(y_pred)
        x, predicted_label = torch.max(y_temp.data, 1)

        val_preds.append(predicted_label.data.cpu().numpy()[0])
        val_preds_prob.append(prob.cpu().numpy()[0])

    ###################### SAVE predictions on TEST set: BEGINS #########################
    with open(PREDICTION_TEST_OUT[:-4] + ".pkl", 'wb') as f:
        pickle.dump(test_preds, f)

    with open(PREDICTION_TEST_PROB_OUT[:-4] + ".pkl", 'wb') as f:
        pickle.dump(test_preds_prob, f)

    f = open(PREDICTION_TEST_OUT, 'w')
    for index, row in df_test.iterrows():
        tweet_id = row['tweet_id']
        gs = row['label']
        test_preds_ = test_preds[index]
        test_preds_prob_ = test_preds_prob[index]

        out = str(tweet_id) + "\t" + str(gs) + "\t" + str(test_preds_prob_) + "\t" + str(test_preds_)
        f.write(out + "\n")
        f.flush()
    f.close()
    print("\nPrediction on test data completed!!!")
    ###################### SAVE predictions on TEST set: ENDS #########################

    ###################### SAVE predictions on VALIDATION set: BEGINS #########################
    with open(PREDICTION_VALID_OUT[:-4] + ".pkl", 'wb') as f:
        pickle.dump(val_preds, f)

    with open(PREDICTION_VALID_PROB_OUT[:-4] + ".pkl", 'wb') as f:
        pickle.dump(val_preds_prob, f)

    f = open(PREDICTION_VALID_OUT, 'w')
    for index, row in df_valid.iterrows():
        tweet_id = row['tweet_id']
        gs = row['label']
        val_preds_ = val_preds[index]
        val_preds_prob_ = val_preds_prob[index]

        out = str(tweet_id) + "\t" + str(gs) + "\t" + str(val_preds_prob_) + "\t" + str(val_preds_)
        f.write(out + "\n")
        f.flush()
    f.close()
    print("Prediction on validation data completed!!!")

    ###################### SAVE predictions on VALIDATION set: ENDS #########################

    # predictions on train data
    train_predictions(model, train_loader_temp, df_train, "train")

    sampling_dict, q2, before_sample, after_sample, sample_status = get_candidacy()

    # get bias on validation set
    PREDS_PATH = PREDICTION_VALID_OUT[:-4] + ".pkl"
    df_main_val = pd.read_csv(MAIN_FILE, sep='\t')
    df_main_val['tweet_id'] = df_main_val['tweet_id'].astype(str)

    B = get_val_bias(VAL_FILE_PATH, PREDS_PATH, df_main_val, q2, "validation")
    print("Bias on validation data: ", B)
    print("Sampled size (N): ", N)
    print("Number of instances duplicated: ", num_duplicated)
    print("Number of instances removed: ", num_removed)

    # get bias on testing set
    PREDS_PATH = PREDICTION_TEST_OUT[:-4] + ".pkl"
    get_test_performance(TEST_FILE_PATH, PREDS_PATH, "testing")

    if B_old is None:
        B_old = B
    else:
        if float(B) == 0.0:
            StopFlag = True
        elif (B_old > 0.0 and B < 0.0) or (B_old < 0.0 and B > 0.0):  # opposite direction
            StopFlag = True
        elif (B > 0.0 and B >= B_old):  # both above 0.0 but increasing
            StopFlag = True
        elif (B < 0.0 and B <= B_old):  # both below 0.0 but decreasing
            StopFlag = True
        else:
            B_old = B

    if StopFlag:

        if (B_old > 0.0 and B < 0.0) or (B_old < 0.0 and B > 0.0):
            if abs(B_old - 0.0) > abs(B - 0.0):
                df_train_last_best = df_train
        df_train_last_best.to_csv(NEW_TRAIN_FILE, sep='\t', index=False)
        sys.exit(0)
    else:
        # set sampled train set as last best
        df_train_last_best = df_train

        # save intermediate training set obtained by sampling
        NEW_TRAIN_FILE_ = NEW_TRAIN_FILE[:-4] + f"_{RUN_NUM}.csv"  # save intermediate train files

        df_train_last_best.to_csv(NEW_TRAIN_FILE_, sep='\t', index=False)

        print("Before sampling: ", before_sample)
        print("Sampling status: ", sample_status)
        print("After sampling: ", after_sample)

        # resample and generate new df_train
        print(f"Sampling {N} train examples")
        df_train = sample_train(df_train, sampling_dict, N)
        main(df_train, df_test, df_valid, StopFlag, RUN_NUM + 1)
    print("*" * 100)


def get_candidacy():
    print("Get candidacy (duplicate/removal)...")

    # check how many removed and duplicated for each group
    sample_status = {'n_ae_nontoxic_duplicated': 0,
                     'n_ae_toxic_removed': 0,
                     'n_super_toxic_duplicated': 0,
                     'n_super_nontoxic_removed': 0}
    before_sample = {'n_ae_nontoxic': 0,
                     'n_ae_toxic': 0,
                     'n_super_toxic': 0,
                     'n_super_nontoxic': 0}

    after_sample = {'n_ae_nontoxic': 0,
                    'n_ae_toxic': 0,
                    'n_super_toxic': 0,
                    'n_super_nontoxic': 0}

    df_all = pd.read_csv(MAIN_FILE, sep='\t')

    # predictions on training dataset
    df_preds = pd.read_csv(PREDICTION_TRAIN_OUT, header=None, sep='\t')
    df_preds.columns = ['tweet_id', 'GS', 'PRED_PROB', 'PRED_LABEL']

    df_all['tweet_id'] = df_all['tweet_id'].astype(str)
    df_preds['tweet_id'] = df_preds['tweet_id'].astype(str)

    # get Pr(toxic) and Pr(nonToxic) and pAE
    df_preds['p_toxic'] = 0.0
    df_preds['p_nonToxic'] = 0.0
    df_preds['pAE'] = 0.0

    df_preds.reset_index(inplace=True, drop=True)

    for index, row in df_preds.iterrows():
        tweet_id = str(row['tweet_id']).replace('n', '')

        preds_prob = row['PRED_PROB'].replace('[', '').replace(']', '').split()
        preds_prob = [float(x) for x in preds_prob]

        df_preds.at[index, 'p_toxic'] = preds_prob[1]
        df_preds.at[index, 'p_nonToxic'] = preds_prob[0]

        df_preds.at[index, 'pAE'] = df_all[df_all['tweet_id'] == tweet_id]['AE'].values[0]

    df_preds.sort_values(by=['pAE'], ascending=True, inplace=True)

    # calculate m(x)
    df_preds['m(x)'] = abs(df_preds['p_toxic'] - df_preds['p_nonToxic'])

    # obtain min, max and median pAE
    pAE = df_preds['pAE'].values.tolist()
    min_pAE = min(pAE)
    max_pAE = max(pAE)
    q2 = np.quantile(pAE, 0.5)

    # get a(x)
    df_preds['a(x)'] = 0.0

    # find the number of intervals
    intervals = 0
    for x in pAE:
        if x < q2 or x > q2:
            intervals += 1

    nfactor = (2.0 / intervals)

    print("min_pAE, q2, max_pAE: ", min_pAE, q2, max_pAE, nfactor)

    # low -1, median 0, high +1
    r = -1

    for index, row in df_preds.iterrows():
        pAE = float(row['pAE'])
        if pAE < q2:
            df_preds.at[index, 'a(x)'] = r
            r += nfactor
        if pAE > q2:
            r += nfactor
            df_preds.at[index, 'a(x)'] = r

    df_preds['|a(x)|'] = abs(df_preds['a(x)'])

    # calculate c(x)
    df_preds['c(x)'] = df_preds['|a(x)|'] - wp * df_preds['m(x)']

    # sort by c(x) in descending order
    df_preds.sort_values(by=['c(x)'], ascending=False, inplace=True)

    df_preds.reset_index(inplace=True, drop=True)

    # determine drop (-1) or duplicate (+1)
    df_preds['sampling'] = 0
    for index, row in df_preds.iterrows():
        pAE = float(row['pAE'])
        GS = row['GS']
        if pAE > q2 and GS == 1:  # AE and toxic
            df_preds.at[index, 'sampling'] = -1
            before_sample['n_ae_toxic'] += 1
        elif pAE > q2 and GS == 0:  # AE and nonToxic
            df_preds.at[index, 'sampling'] = +1
            before_sample['n_ae_nontoxic'] += 1
        elif pAE < q2 and GS == 1:  # nonAE and toxic
            df_preds.at[index, 'sampling'] = +1
            before_sample['n_super_toxic'] += 1
        elif pAE < q2 and GS == 0:  # nonAE and nonToxic
            df_preds.at[index, 'sampling'] = -1
            before_sample['n_super_nontoxic'] += 1
    sampling_dict = dict(zip(df_preds.tweet_id, df_preds.sampling))

    # find number of sample duplicated or removed for each group
    for index, (tweet_id, sampling) in enumerate(sampling_dict.items()):
        if index == 1000: break
        pAE = df_preds[df_preds['tweet_id'] == tweet_id]['pAE'].values[0]
        GS = df_preds[df_preds['tweet_id'] == tweet_id]['GS'].values[0]

        if pAE > q2 and GS == 1:  # AE and toxic
            sample_status['n_ae_toxic_removed'] += 1
        elif pAE > q2 and GS == 0:  # AE and nonToxic
            sample_status['n_ae_nontoxic_duplicated'] += 1
        elif pAE < q2 and GS == 1:  # nonAE and toxic
            sample_status['n_super_toxic_duplicated'] += 1
        elif pAE < q2 and GS == 0:  # nonAE and nonToxic
            sample_status['n_super_nontoxic_removed'] += 1

    # find total toxic and nonToxic for each group
    after_sample['n_ae_toxic'] = before_sample['n_ae_toxic'] - sample_status['n_ae_toxic_removed']
    after_sample['n_super_nontoxic'] = before_sample['n_super_nontoxic'] - sample_status['n_super_nontoxic_removed']

    after_sample['n_ae_nontoxic'] = before_sample['n_ae_nontoxic'] + sample_status['n_ae_nontoxic_duplicated']
    after_sample['n_super_toxic'] = before_sample['n_super_toxic'] + sample_status['n_super_toxic_duplicated']

    return sampling_dict, q2, before_sample, after_sample, sample_status

df_train, df_test, df_valid = load_data()
main(df_train, df_test, df_valid, False, RUN_NUM)