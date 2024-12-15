# https://github.com/jonrusert/robustnessofoffensiveclassifiers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import nltk
from torchtext.legacy.data import Field
from torchtext.legacy.data import TabularDataset
from torchtext.legacy.data import Iterator, BucketIterator

def tokenizer(text):
    FILTER_SIZES = [2, 3, 4]
    token = [t for t in nltk.word_tokenize(text.lower())]
    if len(token) < FILTER_SIZES[-1]:
        for i in range(0, FILTER_SIZES[-1] - len(token)):
            token.append('<pad>')
    return token

class MIDAS:

    def __init__(self, train_data='train.tsv', trained_cnn_model='MIDAS_CNN.pt',
                 trained_blstm_model='MIDAS_BLSTM.pt', trained_blstmGru_model='MIDAS_BLSTM-GRU.pt'):

        # https://torchtext.readthedocs.io/en/latest/data.html#field (pad_token='<pad>', unk_token='<unk>')
        self.TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True)
        self.LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)
        self.ID = Field(sequential=False, use_vocab=False)

        off_datafields = [('id', None), ('text', self.TEXT), ('label', self.LABEL)]

        trn = TabularDataset.splits(path='.', train=train_data, format='tsv', fields=off_datafields)[0]

        self.TEXT.build_vocab(trn, vectors='glove.6B.300d')

        self.BATCH_SIZE = 4

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load pre-trained model
        if torch.cuda.is_available():
            print("Using cuda")
            self.cnn_model = torch.load(trained_cnn_model)
            self.blstm_model = torch.load(trained_blstm_model)
            self.blstmGru_model = torch.load(trained_blstmGru_model)
        else:
            print("Using cpu")
            self.cnn_model = torch.load(trained_cnn_model, map_location='cpu')
            self.blstm_model = torch.load(trained_blstm_model, map_location='cpu')
            self.blstmGru_model = torch.load(trained_blstmGru_model, map_location='cpu')
    
    # prediction for single query
    def predict(self, test_query):
        '''
        predictions using ensemble of three models
        '''

        # CNN
        cnn_votes = {}
        _, cnn_prob = cnn_predict_sentence(self.cnn_model, self.TEXT, self.device, test_query)
        cnn_votes['1'] = cnn_prob

        # BLSTM
        blstm_votes = {}
        _, blstm_prob = blstm_predict_sentence(self.blstm_model, self.TEXT, self.device, test_query)
        blstm_votes['1'] = blstm_prob

        # BLSTM_BGRU
        blstm_gru_votes = {}
        _, blstm_gru_prob = blstm_predict_sentence(self.blstmGru_model, self.TEXT, self.device, test_query)
        blstm_gru_votes['1'] = blstm_gru_prob

        predictions = []
        probs = []

        # ensemble (averaging approach)
        for id in blstm_votes:
            pred = int(round((cnn_votes[id] + blstm_votes[id] + blstm_gru_votes[id]) / 3))
            prob = (cnn_votes[id] + blstm_votes[id] + blstm_gru_votes[id]) / 3

            predictions.append(pred)
            probs.append(prob)

        return predictions[0], probs[0]

    # predictions for multiple queries at once
    def predictMultiple(self, test_queries):

        predictions = []
        probs = []

        for test in test_queries:
            pred, prob = self.predict(test)
            predictions.append(pred)
            probs.append(prob)

        return predictions, probs


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
        # If pad_idx specified, the entries at padding_idx do not contribute to the gradient;
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim), padding=0)  # default padding:0
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, 256)
        self.out = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0) # text = [batch size, sent len]
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))  # cat = [batch size, n_filters * len(filter_sizes)]
        fc1 = self.dropout(self.fc(cat))

        return self.out(fc1)


class BLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden.squeeze(0))


class BLSTM_GRU(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           bidirectional=True)

        self.gru = nn.GRU(hidden_dim * 2, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        # pass to BiGRU
        packed_lstm_outputs = nn.utils.rnn.pack_padded_sequence(output, output_lengths)
        packed_output, (hidden, cell) = self.gru(packed_lstm_outputs)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)        

        return self.fc(output[-1, :, :])

def binary_accuracy(preds, y):
        
    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc


def train_cnn(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, _ = batch.text

        predictions = model(text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_blstm(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths.to('cpu')).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def cnn_predict_sentence(model, TEXT, device, sentence):
    model.eval()
    tokenized = tokenizer(sentence)  # lower cased by tokenizer
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    try:
        prediction = torch.sigmoid(model(tensor))
        return round(prediction.item()), prediction.item()
    except:
        return 0, 0


def blstm_predict_sentence(model, TEXT, device, sentence):
    model.eval()
    tokenized = tokenizer(sentence)  # lower cased by spacy tokenizer
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    try:
        prediction = torch.sigmoid(model(tensor, length_tensor))
        return round(prediction.item()), prediction.item()
    except:
        return 0, 0

def cnn_test(model, iterator):

    model.eval()

    final_preds = []
    pred_probs = []

    with torch.no_grad():

        for batch in iterator:
            text, _ = batch.text

            predictions = model(text).squeeze(1)

            sigmoid_preds = torch.sigmoid(predictions)
            rounded_preds = torch.round(sigmoid_preds)

            # save each prediction with corresponding id to list to write to file later
            for i in range(len(batch.id)):
                final_preds.append((batch.id[i].item(), int(rounded_preds[i].item())))
                pred_probs.append((batch.id[i].item(), sigmoid_preds[i].item()))

    return final_preds, pred_probs


def blstm_test(model, iterator):
    model.eval()

    final_preds = []
    pred_probs = []

    with torch.no_grad():

        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths.to('cpu'))

            sigmoid_preds = torch.sigmoid(predictions)
            rounded_preds = torch.round(sigmoid_preds)

            for i in range(len(batch.id)):
                final_preds.append((batch.id[i].item(), int(rounded_preds[i].item())))
                pred_probs.append((batch.id[i].item(), sigmoid_preds[i].item()))

    return final_preds, pred_probs


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main(trainFile, testFile, train_test, out_file=None):
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True)
    LABEL = Field(sequential=False, use_vocab=False, dtype=torch.float)
    ID = Field(sequential=False, use_vocab=False)

    off_datafields = [('id', None), ('text', TEXT), ('label', LABEL)]
    trn = TabularDataset.splits(path='.', train=trainFile, format='tsv', fields=off_datafields)[0]
    tst_datafields = [('id', ID), ('text', TEXT), ('label', None)]
    tst = TabularDataset(path=testFile, format='tsv', fields=tst_datafields)

    TEXT.build_vocab(trn, vectors='glove.6B.300d')

    BATCH_SIZE = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator = BucketIterator(trn, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.text),
                                    sort_within_batch=True)

    test_iterator = Iterator(tst, batch_size=BATCH_SIZE, device=device, sort_key=lambda x: len(x.text),
                             sort_within_batch=True)

    if (train_test == 'train' or train_test == 'both'):

        # train CNN
        # MIDAS CNN use 256 filters, with 2, 3, 4 as filter_sizes, dropout = 0.3
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 300
        N_FILTERS = 256
        FILTER_SIZES = [2, 3, 4]
        OUTPUT_DIM = 1
        DROPOUT = 0.3
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        optimizer = optim.Adam(model.parameters())

        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 3

        # best_valid_loss = float('inf')

        print('training CNN')
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = train_cnn(model, train_iterator, optimizer, criterion)
            # valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # if valid_loss < best_valid_loss:
            #    best_valid_loss = valid_loss
            #    torch.save(model.state_dict(), 'tut4-model.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            # print(f'\t Valid. Loss: {valid_loss:.3f} |  Valid. Acc: {valid_acc*100:.2f}%')

        torch.save(model, 'MIDAS_CNN.pt')

        # train BLSTM
        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 300
        HIDDEN = 64
        OUTPUT_DIM = 1
        DROPOUT = 0.2
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = BLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN, OUTPUT_DIM, DROPOUT, PAD_IDX)

        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        optimizer = optim.Adam(model.parameters())

        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 3

        # best_valid_loss = float('inf')

        print('training BLSTM')
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = train_blstm(model, train_iterator, optimizer, criterion)
            # valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # if valid_loss < best_valid_loss:
            #    best_valid_loss = valid_loss
            #    torch.save(model.state_dict(), 'tut4-model.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            # print(f'\t Valid. Loss: {valid_loss:.3f} |  Valid. Acc: {valid_acc*100:.2f}%')

        torch.save(model, 'MIDAS_BLSTM.pt')

        # train BLSTM-GRU

        INPUT_DIM = len(TEXT.vocab)
        EMBEDDING_DIM = 300
        HIDDEN = 64
        OUTPUT_DIM = 1
        DROPOUT = 0.3
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = BLSTM_GRU(INPUT_DIM, EMBEDDING_DIM, HIDDEN, OUTPUT_DIM, DROPOUT, PAD_IDX)

        pretrained_embeddings = TEXT.vocab.vectors

        model.embedding.weight.data.copy_(pretrained_embeddings)

        UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        optimizer = optim.Adam(model.parameters())

        criterion = nn.BCEWithLogitsLoss()

        model = model.to(device)
        criterion = criterion.to(device)

        N_EPOCHS = 3

        # best_valid_loss = float('inf')

        print('training BLSTM-GRU')
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            train_loss, train_acc = train_blstm(model, train_iterator, optimizer, criterion)
            # valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # if valid_loss < best_valid_loss:
            #    best_valid_loss = valid_loss
            #    torch.save(model.state_dict(), 'tut4-model.pt')

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            # print(f'\t Valid. Loss: {valid_loss:.3f} |  Valid. Acc: {valid_acc*100:.2f}%')

        torch.save(model, 'MIDAS_BLSTM-GRU.pt')

    if (train_test == 'test' or train_test == 'both'):

        # load three models, get predictions
        # CNN
        if torch.cuda.is_available():
            print("Using cuda")
            model = torch.load('MIDAS_CNN.pt')
            model = model.cuda()
        else:
            print("Using cpu")
            model = torch.load('MIDAS_CNN.pt', map_location='cpu')

        _, cnn_probs = cnn_test(model, test_iterator)

        cnn_votes = {}
        for id, pred in cnn_probs:
            cnn_votes[str(id)] = pred

        # BLSTM
        if torch.cuda.is_available():
            print("Using cuda")
            model = torch.load('MIDAS_BLSTM.pt')
            model = model.to(device)
        else:
            print("Using cpu")
            model = torch.load('MIDAS_BLSTM.pt', map_location='cpu')

        _, blstm_probs = blstm_test(model, test_iterator)
        blstm_votes = {}

        for id, pred in blstm_probs:
            blstm_votes[str(id)] = pred

        # BLSTM_BGRU
        if torch.cuda.is_available():
            print("Using cuda")
            model = torch.load('MIDAS_BLSTM-GRU.pt')
            model = model.to(device)
        else:
            print("Using cpu")
            model = torch.load('MIDAS_BLSTM-GRU.pt', map_location='cpu')

        _, blstm_gru_probs = blstm_test(model, test_iterator)

        blstm_gru_votes = {}

        for id, pred in blstm_gru_probs:
            blstm_gru_votes[str(id)] = pred

        if (out_file):
            output = open(out_file, 'w')
        else:
            output = open('MIDAS_predictionsOut', 'w')

        # ensemble (averaging approach)
        for id in blstm_votes:
            pred = int(round((cnn_votes[id] + blstm_votes[id] + blstm_gru_votes[id]) / 3))

            output.write(str(id) + ',' + str(pred) + '\n')

        output.close()
