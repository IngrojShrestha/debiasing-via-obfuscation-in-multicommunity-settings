import sys

obfuscator = sys.argv[1]

print("Training MIDAS/NULI obfuscator...")

if obfuscator == 'MIDAS':
    from MIDAS import main as obfuscation_train_main

    obfuscation_train_main(
        trainFile="../data/offenseval-training-v1.tsv",
        testFile="/path/to/test.csv",
        train_test="train",
        out_file='MIDAS_out.csv')

elif obfuscator == 'NULI':
    from NULI import main as obfuscation_train_main

    obfuscation_train_main(
        trainFile="../data/offenseval-training-v1.tsv",
        testFile="/path/to/test.csv",
        train_test="train",
        out_file='NULI_out.csv')
    
else:
    print("Invalid obfuscator. Use 'MIDAS' or 'NULI'.")
    sys.exit(0)
