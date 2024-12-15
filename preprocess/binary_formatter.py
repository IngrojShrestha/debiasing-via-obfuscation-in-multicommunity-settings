import pandas as pd
import sys

def convert_to_binary_dwmw17(in_file, out_file):
    df = pd.read_csv(in_file, sep='\t')
    # hate (0), offensive (1), none (2)
    # 0 + 1 -> 1 (hate) and 2 -> 0 (not hate)
    df["label"].replace({0: 1}, inplace=True)
    df["label"].replace({2: 0}, inplace=True)
    df.to_csv(out_file, sep='\t', index=False)


def convert_to_binary_fdcl18(in_file, out_file):
    df = pd.read_csv(in_file, sep='\t')
    # abusive + hateful -> 1 (hate) and spam + normal -> 0 (non hate)
    df["label"].replace({'abusive': 1, 'hateful': 1, 'spam': 0, 'normal': 0}, inplace=True)
    df.to_csv(out_file, sep='\t', index=False)


def convert_to_binary_hateval19(in_file, out_file):
    df = pd.read_csv(in_file, sep='\t')
    # hate speech (1) and non-hate speech (0)
    # no need as it is already binary 1/0
    # df["label"].replace({1: 1}, inplace=True)
    # df["label"].replace({0: 0}, inplace=True)
    df.to_csv(out_file, sep='\t', index=False)

def main(data, in_filename, out_filename):
    if data == 'DWMW17':
        convert_to_binary_dwmw17(in_filename, out_filename)
    elif data == 'FDCL18':
        convert_to_binary_fdcl18(in_filename, out_filename)
    # not required as the dataset is already binary 1/0
    elif data == 'HatEval19':
        convert_to_binary_hateval19(in_filename, out_filename) # just to keep consistent filename for the next step
    else:
        print("Invalid dataset")
        sys.exit(0)

if __name__ == "__main__":
    data = sys.argv[1] # DWMW17, FDCL18, HatEval19
    in_filename = sys.argv[2] # file obtained after dialect estimation
    out_filename = sys.argv[3]  # output filename

    main(data, in_filename, out_filename)
