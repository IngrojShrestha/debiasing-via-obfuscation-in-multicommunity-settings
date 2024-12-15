import sys
import BERT_base as m

STATE = sys.argv[1]
DATA = sys.argv[2]
RUN = sys.argv[3]
OBF_OUT_PATH = sys.argv[4] if len(sys.argv) > 4 else None

parser = m.get_parser()

# Finetune Original BERT-base model (using original toxicity training set)
if STATE == 'original_train':
    args = parser.parse_args([STATE, DATA, RUN])
    m.setup_args_variables(args)
    print("Finetuning original BERT-base model")
    m.main()
    
# Debiased model: Finetune BERT-base model on the original toxicity training set + obfuscated FPs (from validation set)
elif STATE == 'debias_train':
    if not OBF_OUT_PATH:
        print("Please provide file with obfuscated fps for 'debias_train'")
        sys.exit(0)
    args = parser.parse_args([STATE, DATA, RUN, OBF_OUT_PATH])
    m.setup_args_variables(args)
    print("Finetuning on new training dataset")
    m.main()

else:
    print("Invalid STATE. Use 'debias_train' or 'original_train'.")
    sys.exit(0)
