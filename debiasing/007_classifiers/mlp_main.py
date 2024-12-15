import os
import argparse
import sys
import bcolz
import pickle
import time
import random
import pandas as pd
from nltk import PorterStemmer
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import autograd
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, classification_report

##################################################################################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOVE_PATH = "/path/to/glove/" # this folder contains 6B.300.dat, 6B.300_words.pkl, 6B.300_idx.pkl
porter_stemmer = PorterStemmer()
##################################################################################
# model parameters
##################################################################################
MAX_SEQ_LEN = None
VOCAB_SIZE = None
NUM_CLASSES = None
VOCAB = None
tokenizer = None

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
is_padding_idx = False  # Set to True to make the padding index (0) non-trainable in the Embedding layer.
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


##################################################################################
def set_seed(seed: int):
    # https://huggingface.co/transformers/_modules/transformers/trainer_utils.html
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    print("enable seed")
    # random.seed(seed) # prevents random 's' instances selection for sensitivity analysis
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_parser():
    "separate out parser definition in its own function"
    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    obfuscate = subparser.add_parser('obfuscate')
    original_train = subparser.add_parser('original_train')
    debias_train = subparser.add_parser('debias_train')

    obfuscate.add_argument("DATA", help="provide DATA (e.g., DWMW17): ")
    obfuscate.add_argument("RUN", type=int)

    original_train.add_argument("DATA", help="provide DATA (e.g., DWMW17): ")
    original_train.add_argument("RUN", type=int)

    debias_train.add_argument("DATA", help="provide DATA (e.g., DWMW17): ")
    debias_train.add_argument("NUM_INSTANCES", type=int)  # num_instances
    debias_train.add_argument("RUN", type=int)
    debias_train.add_argument("OBF_OUT_PATH", help="path to obfuscated file: obf_out.csv")

    return parser

def setup_args_variables(args):
    global STATE, DATA, RUN, BASE_OUT_PATH, NUM_INSTANCES, OUT_FOLDER
    global MODEL_NAME, PREDICTION_FILE, PREDICTION_PROB_FILE
    global PREDICTION_VALID_FILE, PREDICTION_VALID_PROB_FILE
    global CHECKPOINT_PATH, CHECKPOINT_ALL_PATH
    global TRAIN_FILE_PATH_MAIN, VALID_FILE_PATH, TEST_FILE_PATH
    global SYNTHETIC_INSTANCES_FILE_PATH_NEW

    # 'Original' training to detect bias before debiasing
    if args.command == 'original_train':
        # use set_seed(42) if original_single_seed_run
        STATE = 'ORIGINAL_MULTIPLE_RUNS'
        DATA = str(args.DATA.strip())
        RUN = int(args.RUN)
        
    # obfuscate FPs from validation set generate synthetic non-toxic instances
    elif args.command == 'obfuscate':
        STATE = 'OBFUSCATE'
        DATA = str(args.DATA.strip())
        RUN = int(args.RUN)
    
    # re-train with original + synthetic instances (obfuscated FPs from validation set)
    elif args.command == 'debias_train':
        # use set_seed(42) if single_seed_run
        STATE = 'DEBIAS_SENSITIVITY_ANALYSIS'
        DATA = str(args.DATA.strip())
        RUN = int(args.RUN)
        NUM_INSTANCES = int(args.NUM_INSTANCES)
        SYNTHETIC_INSTANCES_FILE_PATH_NEW = str(args.OBF_OUT_PATH.strip()) # obfuscated instances

    print("*" * 100)
    print("TRAINING and INFERENCES")
    print("RUN #: ", RUN)
    print("STATE: ", STATE)
    print("DATA: ", DATA)
    print("Device; ", DEVICE)
    print(
        f"Epochs: {NUM_EPOCHS}, embedding_dim: {EMBEDDING_DIM}, batch_size: {BATCH_SIZE}, lr: {LEARNING_RATE}, hidden_size: {HIDDEN_SIZE}")
    print(f"l1_reg: {l1_alpha}, l2_reg: {l2_alpha}, dropout: {DROPOUT}")
    print(f"padding: {padding}, truncating: {truncating}")
    print(f"Activation: {ACTIVATION}, is_padding_idx: {is_padding_idx}, UNK_embedding: {UNK_embedding}")
    print(f"early_stopping: {is_early_stopping}, patient: {patience}")
    print(f"max_seq_length: {MAX_SEQ_LEN_type}")
    print(f"is_grad_clip: {is_grad_clip}")
    print("*" * 100)

    BASE_OUT_PATH = "../output/MLP/out/"

    if STATE == 'ORIGINAL_MULTIPLE_RUNS' or STATE == 'OBFUSCATE':
        OUT_FOLDER = f'main_data_prediction_multiple_runs/{DATA}/Run_{RUN}'
        MODEL_NAME = BASE_OUT_PATH + f"{OUT_FOLDER}/mlp_{DATA.lower()}_original.pt"
            
    elif STATE == 'DEBIAS_SENSITIVITY_ANALYSIS':
        OUT_FOLDER = f'debias_obfuscation_sensitivity/{DATA}/{NUM_INSTANCES}/Run_{RUN}'
        MODEL_NAME = BASE_OUT_PATH + f"{OUT_FOLDER}/mlp_{DATA.lower()}_debiased.pt"

    if not os.path.exists(BASE_OUT_PATH + OUT_FOLDER):
        os.makedirs(BASE_OUT_PATH + OUT_FOLDER)

    PREDICTION_FILE = BASE_OUT_PATH + f"{OUT_FOLDER}/predictions_{DATA.lower()}.csv"
    PREDICTION_PROB_FILE = BASE_OUT_PATH + f"{OUT_FOLDER}/predictions_prob_{DATA.lower()}.csv"

    PREDICTION_VALID_FILE = BASE_OUT_PATH + f"{OUT_FOLDER}/valid_predictions_{DATA.lower()}.csv"
    PREDICTION_VALID_PROB_FILE = BASE_OUT_PATH + f"{OUT_FOLDER}/valid_predictions_prob_{DATA.lower()}.csv"

    CHECKPOINT_PATH = BASE_OUT_PATH + f"{OUT_FOLDER}/checkpoint_{DATA.lower()}.pt"
    CHECKPOINT_ALL_PATH = BASE_OUT_PATH + f"{OUT_FOLDER}/checkpoint_all_{DATA.lower()}.pt"

    TRAIN_FILE_PATH_MAIN = f"../data/{DATA}/{DATA.lower()}_train.csv"
    VALID_FILE_PATH = f"../data/{DATA}/{DATA.lower()}_valid.csv"
    TEST_FILE_PATH = f"../data/{DATA}/{DATA.lower()}_test.csv"


##################################################################################

def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def load_data():
    print("\nLoading data (train, val)\n")

    if STATE == 'DEBIAS_SENSITIVITY_ANALYSIS':
        df_train_main = pd.read_csv(TRAIN_FILE_PATH_MAIN, sep="\t")  # has header
        df_synthetic_instances = pd.read_csv(SYNTHETIC_INSTANCES_FILE_PATH_NEW, sep="\t", header=None)  # does not have header
        df_synthetic_instances.columns = ['tweet_id', 'tweet', 'label']
        
        # sample 'NUM_INSTANCES' instances (NUM_INSTANCES = total obfuscated FPs (if not sensitivity analysis))
        # NUM_INSTANCES = [20, 40, , ....., TOTAL_FPs] (for sensitivity analysis)
        random_idx = random.sample([x for x in range(0, df_synthetic_instances.shape[0])], NUM_INSTANCES)
        df_synthetic_instances = df_synthetic_instances.iloc[random_idx]

        print("df_synthetic_instances sampled indices: ", random_idx, len(random_idx))

        df_synthetic_instances.reset_index(inplace=True, drop=True)

        # select only tweet_id, tweet and label from main training file
        # synthetic new instances df has those three columns
        df_train_main = df_train_main[['tweet_id', 'tweet', 'label']]

        # merge two files
        df_train = pd.concat([df_train_main, df_synthetic_instances])
        df_train.reset_index(inplace=True, drop=True)
    else:
        df_train = pd.read_csv(TRAIN_FILE_PATH_MAIN, sep="\t")

    df_test = pd.read_csv(TEST_FILE_PATH, sep="\t")
    df_valid = pd.read_csv(VALID_FILE_PATH, sep="\t")

    # lower tweets
    df_train["tweet"] = df_train["tweet"].apply(lambda x: x.lower())
    df_test["tweet"] = df_test["tweet"].apply(lambda x: x.lower())
    df_valid["tweet"] = df_valid["tweet"].apply(lambda x: x.lower())

    # stemming
    df_train['tweet'] = df_train['tweet'].apply(stem_sentences)
    df_test['tweet'] = df_test['tweet'].apply(stem_sentences)
    df_valid['tweet'] = df_valid['tweet'].apply(stem_sentences)

    return df_train, df_test, df_valid


class OBF_MLP:

    def __init__(self, trained_mlp_model="mlp_original.pt"):

        # used to set MAX_SEQ_LEN, VOCAB_SIZE, NUM_CLASSES and tokenizer
        df_train, df_test, df_valid = load_data()
        
        create_datasets(df_train, df_test, df_valid)

        self.device = DEVICE
        self.tokenizer = tokenizer

        # load pre-trained model
        if torch.cuda.is_available():
            print("Using cuda")

            model = MLP()
            model.load_state_dict(torch.load(trained_mlp_model))

            # run using GPU
            if torch.cuda.is_available():
                print("Using cuda")
                self.model = model.cuda()
            
            # run using CPU
            else:
                print("Using cpu")
                self.model = model

    # predict single query
    def predict(self, tweet):

        self.model.eval()

        with torch.no_grad():
            # pre-process text
            tweet = tweet.lower()
            tweet = stem_sentences(tweet)

            # padding as required (since one word is removed requires padding)
            text_sequence = self.tokenizer.texts_to_sequences([tweet])  # word2idx
            text = pad_sequences(text_sequence, maxlen=MAX_SEQ_LEN, padding=padding, truncating=truncating)

            y_train = torch.tensor([1], dtype=torch.long).to(DEVICE)  # [1] or [0] does not matter here

            x_train = torch.tensor(text, dtype=torch.long).to('cuda')
            data = MyDataset(torch.utils.data.TensorDataset(x_train, y_train))
            data_loader = torch.utils.data.DataLoader(data, shuffle=False)

            for i, (x_batch, y_batch, index) in enumerate(data_loader):
                y_temp, embedRepr = self.model(x_batch)
                y_pred = y_temp.detach()
                m = nn.Softmax(dim=1)
                prob = m(y_pred)
                prob = prob.cpu().numpy()[0]
                p_hate = prob[1]  # probability of hate

                _, predicted_label = torch.max(y_temp.data, 1)
                pred_label = predicted_label.data.cpu().numpy()[0]

                return int(pred_label), p_hate

    # predict multiple queries at once
    def predictMultiple(self, test_queries):

        predictions = []
        probs = []

        for test in test_queries:
            pred, p_hate = self.predict(test)
            predictions.append(pred)
            probs.append(p_hate)

        return predictions, probs

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # load pre-trained glove embeddings
        glove = load_pretrained_glove()
        
        # create embedding matrix
        embedding_matrix = glove_weight_matrix(glove)

        if is_padding_idx:
            self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx=0)
        else:
            self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

        # set weights
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        
        self.embedding.weight.requires_grad = True

        # Linear function 1
        self.fc1 = nn.Linear(EMBEDDING_DIM * MAX_SEQ_LEN, HIDDEN_SIZE)

        # Non-linearity 1
        self.activation = ACTIVATION

        self.dropout = nn.Dropout(p=DROPOUT)

        # Linear function 2
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, text):
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

        return preds, embedRepr

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


class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def create_tokenizer(lines):
    tokenizer = Tokenizer(oov_token=True)
    tokenizer.fit_on_texts(lines)
    return tokenizer


def get_performance(DATA_PATH, PREDS_PATH, dataset):
    '''
    TEST_DATA_PATH, PREDS_PATH, dataset
    '''

    def load_gs_preds():
        with open(PREDS_PATH, 'rb') as f:
            preds = pickle.load(f)

        df = pd.read_csv(DATA_PATH, sep='\t')
        y_true = df['label'].values.tolist()

        y_pred = []
        for ps in preds:
            y_pred.append(ps)

        return y_true, y_pred

    def load_test_data():
        df = pd.read_csv(DATA_PATH, sep="\t")
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

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        TPR, FPR, FNR, PPV, NPV = compute_eval_metrics(tn, fp, fn, tp)

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = round(accuracy_score(y_true, y_pred), 3)

        print(classification_report(y_true, y_pred, digits=4))
        print("Accuracy/microF1: ", accuracy)
        print("macro_f1: ", round(macro_f1, 3))
        print("\nConfusuion matrix: ")
        print(f"TP: {tp}, FP: {fp} FN: {fn} TN: {tn}")
        print("\n Fairness metrics: ")
        print(TPR, FPR, FNR, NPV)

    def eval_performance_main():
        print("*" * 10 + f"Bias on {dataset} set begins" + "*" * 10)
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
        print(f"TP: {TPs}, FP: {FPs} FN: {FNs} TN: {TNs}")

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        print(classification_report(y_true, y_pred, digits=4))

        print("macro_f1: ", round(macro_f1, 3))
        print_confusion_matrix_group(df_all, "ae")
        print_confusion_matrix_group(df_all, "white")
        print_confusion_matrix_group(df_all, 'hispanic')
        print_confusion_matrix_group(df_all, 'asian')

        print("*" * 10 + f"Bias on {dataset} set ends" + "*" * 10)
        print("\n")

    eval_performance_main()

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
    from collections import Counter

    lengths = sorted(lengths)

    dist = dict(Counter(lengths))

    temp_dist = dict()

    for k, v in dist.items():
        if v >= 5:
            temp_dist[k] = v

    return max(list(temp_dist.keys()))


def create_datasets(df_train, df_test, df_valid):
    global VOCAB_SIZE, NUM_CLASSES, VOCAB, MAX_SEQ_LEN, tokenizer

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

    print("\nLabels: ", df_train['label'].unique())
    print(f"MAX_SEQ_LEN : {MAX_SEQ_LEN}, VOCAB_SIZE: {VOCAB_SIZE}, NUM_CLASSES: {NUM_CLASSES}")

    print("\nConverting train data to sequence...")
    train_sequences = tokenizer.texts_to_sequences(df_train['tweet'].values)  # word2idx
    train_x = pad_sequences(train_sequences, maxlen=MAX_SEQ_LEN, padding=padding, truncating=truncating)  # padding
    train_y = df_train['label'].values

    print("\nConverting valid data to sequence...")
    valid_sequences = tokenizer.texts_to_sequences(df_valid['tweet'].values)  # word2idx
    valid_x = pad_sequences(valid_sequences, maxlen=MAX_SEQ_LEN, padding=padding, truncating=truncating)  # padding
    valid_y = df_valid['label'].values

    print("\nConverting test data to sequence...\n")
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
    valid_loader_unbatched = torch.utils.data.DataLoader(valid, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, shuffle=False)

    return train_loader, valid_loader, valid_loader_unbatched, test_loader


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


def l1_penalty(params, l1_lambda=l1_alpha):
    """Returns the L1 penalty of the params."""
    l1_norm = sum(p.abs().sum() for p in params)
    return l1_lambda * l1_norm


def l2_penalty(params, l2_lambda=l2_alpha):
    l2_reg = torch.tensor(0.)
    for param in params:
        l2_reg += torch.norm(param.to('cpu'))
    return l2_lambda * l2_reg


def train_model(model, optimizer, criterion, patience, train_loader, valid_loader):
    # Track training and validation losses during training
    train_losses = []        # Training loss per batch
    valid_losses = []        # Validation loss per batch
    avg_train_losses = []    # Average training loss per epoch
    avg_valid_losses = []    # Average validation loss per epoch

    get_current_date_time(status="Training Start")

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
                outputs, embed_repr = model(x_batch)

                m = nn.Softmax(dim=1)

                prob = m(outputs)
                x, predicted_label = torch.max(outputs.data, 1)

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
                outputs, embed_repr = model(x_batch)
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

    get_current_date_time(status="Training End")

    print("Time taken to train the model: " + str(time.time() - start_time))

    return model, avg_train_losses, avg_valid_losses, optimizer


def visualize_loss_early_stopping(train_loss, valid_loss):
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
    # plt.show()
    fig.savefig(BASE_OUT_PATH + OUT_FOLDER + "/" + f'loss_plot_{DATA.lower()}.png', bbox_inches='tight')


def main():
    df_train, df_test, df_valid = load_data()

    train_loader, valid_loader, valid_loader_unbatched, test_loader = create_datasets(df_train, df_test, df_valid)

    # TRAIN
    print(f"\nTraining model {STATE}...\n")

    # Initialized MLP
    model = MLP()
    print("Model: ", model)

    if torch.cuda.is_available():
        model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: ", total_params)

    # L2 regularization: weight_decay
    if enable_l2:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=l2_alpha)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = torch.nn.CrossEntropyLoss()

    model, train_loss, valid_loss, optimizer = train_model(model, optimizer, criterion, patience, train_loader, valid_loader)  

    visualize_loss_early_stopping(train_loss, valid_loss)

    # save model
    torch.save(model.state_dict(), MODEL_NAME)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, CHECKPOINT_ALL_PATH)

    model = MLP()
    model = model.cuda()
    model.load_state_dict(torch.load(MODEL_NAME))

    model.eval()

    ########################################################################

    test_preds = []
    test_preds_prob = []

    # predict all samples in the test set batch per batch
    for i, (x_batch, y_batch, index) in enumerate(test_loader):
        y_temp, embed_repr = model(x_batch)
        y_pred = y_temp.detach()

        m = nn.Softmax(dim=1)
        prob = m(y_pred)
        x, predicted_label = torch.max(y_temp.data, 1)

        test_preds.append(predicted_label.data.cpu().numpy()[0])
        test_preds_prob.append(prob.cpu().numpy()[0])

    print("\nStoring predictions on test set...")

    with open(PREDICTION_FILE[:-4] + ".pkl", 'wb') as f:
        pickle.dump(test_preds, f)

    with open(PREDICTION_PROB_FILE[:-4] + ".pkl", 'wb') as f:
        pickle.dump(test_preds_prob, f)

    f = open(PREDICTION_FILE, 'w')

    for index, row in df_test.iterrows():
        # print(f"Prediction :{index + 1} out of {df_test.shape[0]}")
        tweet_id = row['tweet_id']
        gs = row['label']
        test_preds_ = test_preds[index]
        test_preds_prob_ = test_preds_prob[index]

        out = str(tweet_id) + "\t" + str(gs) + "\t" + str(test_preds_prob_) + "\t" + str(test_preds_)
        f.write(out + "\n")

    PREDICTION_FILE_ = PREDICTION_FILE[:-4] + ".pkl"
    get_performance(DATA_PATH=TEST_FILE_PATH, PREDS_PATH=PREDICTION_FILE_, dataset="testing")

    ########################################################################

    valid_preds = []
    valid_preds_prob = []

    # predict all samples in the test set batch per batch
    for i, (x_batch, y_batch, index) in enumerate(valid_loader_unbatched):
        y_temp, embed_repr = model(x_batch)
        y_pred = y_temp.detach()

        m = nn.Softmax(dim=1)
        prob = m(y_pred)
        x, predicted_label = torch.max(y_temp.data, 1)

        valid_preds.append(predicted_label.data.cpu().numpy()[0])
        valid_preds_prob.append(prob.cpu().numpy()[0])

    print("\nStoring predictions on valid set...")

    with open(PREDICTION_VALID_FILE[:-4] + ".pkl", 'wb') as f:
        pickle.dump(valid_preds, f)

    with open(PREDICTION_VALID_PROB_FILE[:-4] + ".pkl", 'wb') as f:
        pickle.dump(valid_preds_prob, f)

    f = open(PREDICTION_VALID_FILE, 'w')

    for index, row in df_valid.iterrows():
        # print(f"Prediction :{index + 1} out of {df_test.shape[0]}")
        tweet_id = row['tweet_id']
        gs = row['label']
        valid_preds_ = valid_preds[index]
        valid_preds_prob_ = valid_preds_prob[index]

        out = str(tweet_id) + "\t" + str(gs) + "\t" + str(valid_preds_prob_) + "\t" + str(valid_preds_)
        f.write(out + "\n")

    PREDICTION_VALID_FILE_ = PREDICTION_VALID_FILE[:-4] + ".pkl"
    get_performance(DATA_PATH=VALID_FILE_PATH, PREDS_PATH=PREDICTION_VALID_FILE_, dataset="validation")

    print(f"COMPLETED {STATE}!!!")


if __name__ == '__main__':
    main()
