# Debiasing by obfuscating with 007-classifiers promotes fairness in multi-community settings

(To appear in COLING 2025)

## Datasets
HatEval19: Request the dataset here http://hatespeech.di.unito.it/hateval.html
(HS Label: 1, 0)

FDCL18: Request data from the authors
(Label: abusive, normal, hateful, spam)

DWMW17: https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data
(Label: 0 - hate speech, 1 - offensive language, 2 - neither)

## Generating train valid test files
Step 1: Estimate the racial dialect.<br/>
```
python dialect_estimation.py <DATA> <raw_data_file> <output_file_dialect_estimation>
```
Step 2: Modify label to binary (toxic, non-toxic). <br/>
```
python binary_formatter.py <output_file_dialect_estimation> <output_filename_binary_formatter>
```
Step 3: Generate train, valid and test files.
```
python train_valid_test_split.py <output_filename_binary_formatter>
```
## 'Original' bias (MLP)

```
python starter_train_mlp.py original_train <DATA> <RUN>

python starter_train_mlp.py original_train DWMW17 1
python starter_train_mlp.py original_train DWMW17 2
...
python starter_train_mlp.py original_train DWMW17 10
```

## 'Original' bias (BERT-base)

```
python starter_finetune_bert_base.py original_train <DATA> <RUN>

python starter_finetune_bert_base.py original_train DWMW17 1
python starter_finetune_bert_base.py original_train DWMW17 2
...
python starter_finetune_bert_base.py original_train DWMW17 10
```

## Text Classifier as 007-classifier for debiasing: OBF_TC (e.g., TC: MLP or BERT_base)

Step 1: Load original MLP/BERT saved model (e.g. mlp_original.pt) and obfuscate false positive instances (from validation set)

_Run "starter_train_mlp"/"starter_finetune_bert_base" with STATE "original_train" to save original model before loading it_

_Set the required parameters (train file path and original model path) in Obfuscator constructor_

```
python starter_obfuscation.py MLP obfuscate DWMW17 1
python starter_obfuscation.py BERT_base
```

Step 2: Re-train on original + synthetic instances (obfuscated version of false positive instances)

```
# obf_out.csv: contains the obfuscated versions of false positives (FPs) from the validation set (using TC as 007-classifier)
# NUMBER_FPs: for sensitivity analysis (not required for BERT_base)

python starter_train_mlp.py debias_train <DATA> <NUMBER_FPs> <RUN> </path/to/obf_out.csv>
python starter_finetune_bert_base.py debias_train <DATA> <RUN> </path/to/obf_out.csv>

python starter_train_mlp.py debias_train DWMW17 20 1 </path/to/obf_out.csv> 
python starter_train_mlp.py debias_train DWMW17 40 1 </path/to/obf_out.csv>
...
python starter_train_mlp.py debias_train DWMW17 20 2 </path/to/obf_out.csv>
python starter_train_mlp.py debias_train DWMW17 40 2 </path/to/obf_out.csv>
...
```

## MIDAS/NULI as 007-classifier for debiasing: OBF_MIDAS/OBF_NULI

Step 1: Train MIDAS or NULI using the 'offenseval-training-v1.tsv' dataset. For MIDAS, use labels 1 and 0; for NULI, use labels HATE and NOT. Save the trained model.

```
python starter_train_obfuscator.py <MIDAS/NULI>
```

Step 2: Load MIDAS/NULI model and then obfuscate false positive instances (from validation set)

_Set the required parameters (train file path and original model path) in Obfuscator constructor (Obfuscator.py)_

```
python starter_obfuscation.py <MIDAS/NULI>
```

Step3: Re-train on original + synthetic instances (obfuscated version of false positive instances)

```
# obf_out.csv: contains the obfuscated versions of false positives (FPs) from the validation set (using MIDAS/NULI as 007-classifier)

python starter_train_mlp.py debias_train <DATA> <NUMBER_FPs> <RUN> </path/to/obf_out.csv>
python starter_finetune_bert_base.py debias_train <DATA> <RUN> </path/to/obf_out.csv>

python starter_train_mlp.py debias_train DWMW17 20 1 </path/to/obf_out.csv>
...
python starter_train_mlp.py debias_train DWMW17 110 10 </path/to/obf_out.csv>
```

## Sentiment Classifier as 007-classifier for debiasing: OBF_SC
(No need to train obfuscator as we are using off-the-shelf sentiment classifier (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment))

Step 1: Load off-the-shelf sentiment classifier and then obfuscate false positive instances (from validation set)

```
python starter_obfuscation.py <SC>
```

Step2: Re-train on original + synthetic instances (obfuscated version of false positive instances)

```
# obf_out.csv: contains the obfuscated versions of false positives (FPs) from the validation set (using SC as 007-classifier)

python starter_train_mlp.py debias_train <DATA> <NUMBER_FPs> <RUN> </path/to/obf_out.csv>
python starter_finetune_bert_base.py debias_train <DATA> <RUN> </path/to/obf_out.csv>

python starter_train_mlp.py debias_train DWMW17 20 1 </path/to/obf_out.csv>
...
python starter_train_mlp.py debias_train DWMW17 110 10 </path/to/obf_out.csv>
```


## Baseline: DiffT_AE
```
python DiffT_AE.py <DATA> <wp>

wp ={0.1,0.32,1,3.2,10}
```


```
STATE = {original_train, obfuscate, debias_train}
DATA = {DWMW17, HatEval19, FDCL18}
RUN = {1, 2, 3, ..., 10}
```