import sys
import mlp_main as m

STATE = sys.argv[1]
DATA = sys.argv[2]
RUN = sys.argv[3]

# required for 'debias_train'
NUM_INSTANCES = sys.argv[4] if len(sys.argv) > 4 else None
OBF_OUT_PATH = sys.argv[5] if len(sys.argv) > 5 else None 

# Train Original MLP model (using original toxicity dataset)
if STATE == "original_train":
    print("Training original MLP model")
    parser = m.get_parser()
    args = parser.parse_args([STATE, DATA, RUN])
    m.setup_args_variables(args)
    m.main()

# Debiased model: Train MLP model on the original toxicity dataset + obfuscated FPs (from validation set)
elif STATE == "debias_train":
    NUM_INSTANCES = sys.argv[4]
    OBF_OUT_PATH = sys.argv[5]
    
    if NUM_INSTANCES is None:
        print("NUM_INSTANCES is required for 'debias_train'.")
        sys.exit(0)
        
    if OBF_OUT_PATH is None:
        print("Please provide file with obfuscated fps for 'debias_train'.")
        sys.exit(1)
        
    print("Re-training on new training dataset")
    parser = m.get_parser()
    args = parser.parse_args([STATE, DATA, NUM_INSTANCES, RUN, OBF_OUT_PATH])
    m.setup_args_variables(args)
    m.main()

else:
    print("Invalid STATE. Use 'original_train' or 'debias_train'.")
    sys.exit(0)
