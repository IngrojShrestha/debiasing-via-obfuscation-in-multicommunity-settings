import random
import os
import shutil
import logging
import functools
import argparse
import sys
import wandb
import pandas as pd
import pickle
from nltk import PorterStemmer
from sklearn.metrics import f1_score
import sklearn.metrics
from sklearn.metrics import recall_score, confusion_matrix, accuracy_score, classification_report
import torch
from simpletransformers.classification import ClassificationModel

print = functools.partial(print, flush=True)

# Setup logging to capture training progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
porter_stemmer = PorterStemmer()

def get_parser():
    "separate out parser definition in its own function"
    # https://towardsdatascience.com/a-simple-guide-to-command-line-arguments-with-argparse-6824c30ab1c3
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    original_train = subparser.add_parser('original_train')
    debias_train = subparser.add_parser('debias_train')

    original_train.add_argument("DATA", help="provide DATA (e.g., DWMW17): ")
    original_train.add_argument("RUN", type=int)

    debias_train.add_argument("DATA", help="provide DATA (e.g., DWMW17): ")
    debias_train.add_argument("RUN", type=int)
    debias_train.add_argument("OBF_OUT_PATH", help="path to obfuscated file: obf_out.csv")

    return parser

def setup_args_variables(args):
    global STATE, DATA, RUN, BASE_OUT_PATH, OUT_FOLDER
    global MODEL_NAME, TEST_PREDICTION_FILE, RAW_TEST_PREDICTION_FILE
    global VALID_PREDICTION_FILE, RAW_VALID_PREDICTION_FILE
    global CHECKPOINT_ALL_PATH
    global TRAIN_FILE_PATH_MAIN, VALID_FILE_PATH, TEST_FILE_PATH
    global SYNTHETIC_INSTANCES_FILE_PATH_NEW
    global TRAIN_MODEL_PATH

    # 'Original' training to detect bias before debiasing
    if args.command == 'original_train':
        STATE = 'ORIGINAL_MULTIPLE_RUNS'
        DATA = str(args.DATA.strip())
        RUN = int(args.RUN)
    
    # re-train with original + synthetic instances (obfuscated FPs from validation set)
    elif args.command == 'debias_train':
        STATE = 'DEBIAS_RUNS'
        DATA = str(args.DATA.strip())
        RUN = int(args.RUN)
        SYNTHETIC_INSTANCES_FILE_PATH_NEW = str(args.OBF_OUT_PATH.strip()) # obfuscated instances
    
    BASE_OUT_PATH = "../output/bert_base/"

    if STATE == 'ORIGINAL_MULTIPLE_RUNS':
        # for storing prediction on testing and validation set
        OUT_FOLDER = f'main_data_prediction_multiple_runs/{DATA}/run{RUN}'
        
        MODEL_NAME = BASE_OUT_PATH + f"{OUT_FOLDER}/bert_main_{DATA.lower()}.pt"
        
        # for storing the model checkpoints and the best model during training
        TRAIN_MODEL_PATH = BASE_OUT_PATH + f"main_model/{DATA}/outputs/"
        
    elif STATE == 'DEBIAS_RUNS':
        # for storing prediction on testing and validation set
        OUT_FOLDER = f'debias_data_prediction_multiple_runs/{DATA}/run{RUN}'
        
        MODEL_NAME = BASE_OUT_PATH + f"{OUT_FOLDER}/bert_debiased_{DATA.lower()}.pt"
        
        # for storing the model checkpoints and the best model during training
        TRAIN_MODEL_PATH = BASE_OUT_PATH + f"debiased_model/{DATA}/outputs/"

    if not os.path.exists(BASE_OUT_PATH + OUT_FOLDER):
        os.makedirs(BASE_OUT_PATH + OUT_FOLDER)
    
    print("Creating directory to store best models of each runs...")
    if not os.path.exists(TRAIN_MODEL_PATH[:-1] + "_runs"):
        os.makedirs(TRAIN_MODEL_PATH[:-1] + "_runs")

    TEST_PREDICTION_FILE = BASE_OUT_PATH + f"{OUT_FOLDER}/test_predictions_{DATA.lower()}.csv"
    RAW_TEST_PREDICTION_FILE = BASE_OUT_PATH + f"{OUT_FOLDER}/test_raw_predictions_{DATA.lower()}.csv"

    VALID_PREDICTION_FILE = BASE_OUT_PATH + f"{OUT_FOLDER}/valid_predictions_{DATA.lower()}.csv"
    RAW_VALID_PREDICTION_FILE = BASE_OUT_PATH + f"{OUT_FOLDER}/valid_raw_predictions_{DATA.lower()}.csv"

    CHECKPOINT_ALL_PATH = BASE_OUT_PATH + f"{OUT_FOLDER}/checkpoint_all_{DATA.lower()}.pt"

    TRAIN_FILE_PATH_MAIN = f"../data/{DATA}/{DATA.lower()}_train.csv"
    VALID_FILE_PATH = f"../data//{DATA}/{DATA.lower()}_valid.csv"
    TEST_FILE_PATH = f"../data/{DATA}/{DATA.lower()}_test.csv"

def get_bias(DATA_PATH, PREDS_PATH, dataset):
    '''
    TEST_DATA_PATH, PREDS_PATH, dataset
    '''

    def load_gs_preds():
        # print("load_gs_preds\n")
        with open(PREDS_PATH, 'rb') as f:
            preds = pickle.load(f)

        df = pd.read_csv(DATA_PATH, sep='\t')
        y_true = df['label'].values.tolist()

        y_pred = []
        for ps in preds:
            y_pred.append(ps)

        return y_true, y_pred

    def load_test_data():
        # print("load_test_data\n")
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

        binary_f1 = f1_score(y_true, y_pred, average='binary')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        accuracy = round(accuracy_score(y_true, y_pred), 3)

        print(classification_report(y_true, y_pred, digits=4))

        print("Accuracy/microF1: ", accuracy)
        print("binary_f1: ", round(binary_f1, 3))
        print("macro_f1: ", round(macro_f1, 3))
        print("micro_f1: ", round(micro_f1, 3))
        print("weighted_f1: ", round(weighted_f1, 3))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("\nConfusuion matrix: ")
        print("TP: ", tp)
        print("FP: ", fp)
        print("FN: ", fn)
        print("TN: ", tn)

        print("\n Fairness metrics: ")
        TPR, FPR, FNR, PPV, NPV = compute_eval_metrics(tn, fp, fn, tp)

        print("\nFormatted\n")
        overall_cf = "\makecell[l]{" + str(tp) + "," + str(fp) + "," + str(fn) + "," + str(tn) + "}"

        out = str(TPR) + " & " + str(FPR) + " & " + str(FNR) + " & " + str(PPV) + " & " + str(NPV) + " & " + \
              str(accuracy) + " & " + str(round(binary_f1, 3)) + " & " + str(round(macro_f1, 3)) + \
              " & " + str(round(weighted_f1, 3)) + " & " + overall_cf

        print(out)

    def test_main():
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
        print("Overall confusion matrix (all groups- sub, super, excluded)")
        TPs = df_all[(df_all['gs'] == 1) & (df_all['preds'] == 1)].shape[0]
        TNs = df_all[(df_all['gs'] == 0) & (df_all['preds'] == 0)].shape[0]
        FNs = df_all[(df_all['gs'] == 1) & (df_all['preds'] == 0)].shape[0]
        FPs = df_all[(df_all['gs'] == 0) & (df_all['preds'] == 1)].shape[0]
        print("TP: ", TPs)
        print("FP: ", FPs)
        print("FN: ", FNs)
        print("TN: ", TNs)
        binary_f1 = f1_score(y_true, y_pred, average='binary')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')

        print(classification_report(y_true, y_pred, digits=4))

        overall_cf = "\makecell[l]{" + str(TPs) + "," + str(FPs) + "," + str(FNs) + "," + str(TNs) + "}"
        print("Overall CF: ", overall_cf)

        print("Accuracy/microF1: ", round(accuracy_score(y_true, y_pred), 3))
        print("binary_f1: ", round(binary_f1, 3))
        print("macro_f1: ", round(macro_f1, 3))
        print("micro_f1: ", micro_f1)
        print("weighted_f1: ", round(weighted_f1, 3))
        out = str(round(accuracy_score(y_true, y_pred), 3)) + " & " + str(round(binary_f1, 3)) + \
              " & " + str(round(macro_f1, 3)) + " & " + str(round(weighted_f1, 3)) + " & " + overall_cf
        print(out)
        
        print_confusion_matrix_group(df_all, "ae")
        print_confusion_matrix_group(df_all, "white")
        print_confusion_matrix_group(df_all, 'hispanic')
        print_confusion_matrix_group(df_all, 'asian')

        print("*" * 10 + f"Bias on {dataset} set ends" + "*" * 10)
        print("\n")

    test_main()
    

def load_data():
    print("\nLoading data (train, val)\n")

    if STATE == 'DEBIAS_RUNS':
        df_train_main = pd.read_csv(TRAIN_FILE_PATH_MAIN, sep="\t")  # has header
        
        # select only tweet_id, tweet and label from main training file
        df_train_main = df_train_main[['tweet_id', 'tweet', 'label']]
        df_train_main.reset_index(inplace=True, drop=True)
    
        # augmenting data
        df_synthetic_instances = pd.read_csv(SYNTHETIC_INSTANCES_FILE_PATH_NEW, sep="\t", header=None)  # does not have header
        df_synthetic_instances.columns = ['tweet_id', 'tweet', 'label']
    
        # merge two files
        df_train = pd.concat([df_train_main, df_synthetic_instances])
        df_train.reset_index(inplace=True, drop=True)
    else:
        df_train = pd.read_csv(TRAIN_FILE_PATH_MAIN, sep="\t") # has header
        df_train = df_train[['tweet_id', 'tweet', 'label']]
        df_train.reset_index(inplace=True, drop=True)

    df_test = pd.read_csv(TEST_FILE_PATH, sep="\t")
    df_valid = pd.read_csv(VALID_FILE_PATH, sep="\t")
    df_valid_eval = pd.read_csv(VALID_FILE_PATH, sep="\t")

    # lower tweets for consistency
    df_train["tweet"] = df_train["tweet"].apply(lambda x: x.lower())
    df_test["tweet"] = df_test["tweet"].apply(lambda x: x.lower())
    df_valid["tweet"] = df_valid["tweet"].apply(lambda x: x.lower())
    df_valid_eval["tweet"] = df_valid_eval["tweet"].apply(lambda x: x.lower())

    df_train= df_train.drop(columns=['tweet_id']) 
    df_train= df_train.rename(columns={"tweet": "text", "label": "labels"})
    
    df_valid_eval= df_valid_eval.drop(columns=['tweet_id']) 
    df_valid_eval= df_valid_eval.rename(columns={"tweet": "text", "label": "labels"})
    
    return df_train, df_test, df_valid, df_valid_eval

def f1_macro(labels, preds):

    #e.g., results = {'mcc': 0.4582553866223619, 'tp': 471, 'tn': 964, 'fp': 158, 'fn': 348, 'auroc': 0.8206657177245413, 'auprc': 0.7602741286164156, 'f1': 0.7213321181966343, 'eval_loss': 0.5278772535871287}
    
    preds_tensor = torch.tensor(preds)
    if preds_tensor.dim() == 1 or preds_tensor.size(1) == 1:
        preds = (preds_tensor > 0.5).int().numpy()
    else:  
        preds = torch.argmax(preds_tensor, axis=1).numpy()
        
    macro_f1_score = sklearn.metrics.f1_score(labels, preds, average='macro')
    
    # print("Calculated macro F1:", macro_f1_score, flush= True)
    # return {'macro_f1': macro_f1_score}
    return macro_f1_score

def main():
    df_train, df_test, df_valid, df_valid_eval = load_data()

########################################################################
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # https://simpletransformers.ai/docs/usage/
    '''
    args = {
       'output_dir': 'outputs/',
       'cache_dir': 'cache/',
       'fp16': True,
       'fp16_opt_level': 'O1',
       'max_seq_length': 256,
       'train_batch_size': 8,
       'eval_batch_size': 8,
       'gradient_accumulation_steps': 1,
       'num_train_epochs': 3,
       'weight_decay': 0,
       'learning_rate': 4e-5,
       'adam_epsilon': 1e-8,
       'warmup_ratio': 0.06,
       'warmup_steps': 0,
       'max_grad_norm': 1.0,
       'logging_steps': 50,
       'evaluate_during_training': False,
       'save_steps': 2000,
       'eval_all_checkpoints': True,
       'use_tensorboard': True,
       'overwrite_output_dir': True,
       'reprocess_input_data': False,
    }
    '''
    
    # https://docs.wandb.ai/quickstart/
    wandb.login(key="your-wandb-api-key")
    
    total_train_samples = len(df_train) 
    batch_size = 32 
    steps_per_epoch = total_train_samples // batch_size  # Calculate steps per epoch

    # Create a ClassificationModel
    # https://simpletransformers.ai/docs/usage/
    model = ClassificationModel(
        model_type= 'bert', model_name='bert-base-uncased',
        args={
            'do_lower_case': True,
            'reprocess_input_data': True, 
            'overwrite_output_dir': True,
            'learning_rate': 2e-5,
            'max_seq_length': 128, 
            'train_batch_size': 32,
            'eval_batch_size':32,
            'num_train_epochs': 10, 
            'evaluate_during_training': True, 
            'evaluate_during_training_verbose': True,
            'use_early_stopping': True, 
            # 'early_stopping_metric': "f1",
            # 'early_stopping_metric_minimize': False, # set to False as a higher macro F1 score indicates better model performance (goal: maximizae)
            'early_stopping_metric': "eval_loss",
            'early_stopping_metric_minimize': True, # set to True as we want to minimize validation loss
            'early_stopping_consider_epochs': True,
            'early_stopping_patience': 3, 
            'early_stopping_delta': 0,
            'evaluate_during_training_steps': steps_per_epoch,
            'manual_seed': None,
            'wandb_project': f'{DATA.lower()}-bert-base-obf-bert-no-seed',
            'output_dir': TRAIN_MODEL_PATH,
            'best_model_dir':TRAIN_MODEL_PATH + "best_model"
        },
        use_cuda=True
    )

    model.train_model(df_train, eval_df=df_valid_eval, f1=f1_macro)

    # load best_model
    best_model = ClassificationModel(
        model_type='bert',
        model_name=TRAIN_MODEL_PATH + "best_model",
        use_cuda=True
    )
    
    # Save the best model manually
    torch.save(best_model, MODEL_NAME)
    
    ########################################################################
    print("\nStoring predictions on test set...")
    predictions_test, raw_outputs_test = best_model.predict(df_test.tweet.tolist())
    
    with open(TEST_PREDICTION_FILE[:-4] + ".pkl", 'wb') as f:
        pickle.dump(predictions_test, f)
        
    with open(RAW_TEST_PREDICTION_FILE[:-4] + ".pkl", 'wb') as f:
        pickle.dump(raw_outputs_test, f)
        
    f = open(TEST_PREDICTION_FILE, 'w')
    f.write("tweet_id" + "\t" + "gs" + "\t" + "pred_prob" + "\t" + "preds" + "\n")

    for index, row in df_test.iterrows():
        # print(f"Prediction :{index + 1} out of {df_test.shape[0]}")
        tweet_id = row['tweet_id']
        gs = row['label']
        test_preds_ = predictions_test[index]
        test_preds_prob_ = raw_outputs_test[index]

        out = str(tweet_id) + "\t" + str(gs) + "\t" + str(test_preds_prob_) + "\t" + str(test_preds_)
        f.write(out + "\n")
        f.flush()
        os.fsync(f)
    f.close()
    
    TEST_PREDICTION_FILE_ = TEST_PREDICTION_FILE[:-4] + ".pkl"
    get_bias(DATA_PATH=TEST_FILE_PATH, PREDS_PATH=TEST_PREDICTION_FILE_, dataset="testing")
    
    ########################################################################
    
    print("\nStoring predictions on valid set...")
    predictions_valid, raw_outputs_valid = best_model.predict(df_valid.tweet.tolist())
    
    with open(VALID_PREDICTION_FILE[:-4] + ".pkl", 'wb') as f:
        pickle.dump(predictions_valid, f)
        
    with open(RAW_VALID_PREDICTION_FILE[:-4] + ".pkl", 'wb') as f:
        pickle.dump(raw_outputs_valid, f)
        
    f = open(VALID_PREDICTION_FILE, 'w')
    f.write("tweet_id" + "\t" + "gs" + "\t" + "pred_prob" + "\t" + "preds" + "\n")

    for index, row in df_valid.iterrows():
        # print(f"Prediction :{index + 1} out of {df_test.shape[0]}")
        tweet_id = row['tweet_id']
        gs = row['label']
        valid_preds_ = predictions_valid[index]
        valid_preds_prob_ = raw_outputs_valid[index]

        out = str(tweet_id) + "\t" + str(gs) + "\t" + str(valid_preds_prob_) + "\t" + str(valid_preds_)
        f.write(out + "\n")
        f.flush()
        os.fsync(f)
    f.close()
    
    VALID_PREDICTION_FILE_ = VALID_PREDICTION_FILE[:-4] + ".pkl"
    get_bias(DATA_PATH=VALID_FILE_PATH, PREDS_PATH=VALID_PREDICTION_FILE_, dataset="validation")
    
    ########################################################################
    # For multiple runs, rename the outputs/ folder (storing checkpoints and the best model) to outputs_runN (N = 1 to 10),
    # and move the renamed folder to the outputs_runs/ directory.
    os.rename(TRAIN_MODEL_PATH, TRAIN_MODEL_PATH[:-1] + f"_run_{RUN}" + "/")
    shutil.move(TRAIN_MODEL_PATH[:-1] + f"_run_{RUN}", TRAIN_MODEL_PATH[:-1] + "_runs")
    
    print(f"Renamed and moved run {RUN} best model!!")

if __name__ == '__main__':
    main()
