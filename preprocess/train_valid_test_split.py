import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_sample(data, in_filename):
    df = pd.read_csv(in_filename, sep='\t')

    # filter tweets with None (group:-1) as dialect
    df = df[df['group'] != -1].reset_index()

    # use tweet_processed for training classifier.
    # so, we remove original tweet (col: tweet) and rename column tweet_processed to tweet
    df.drop(columns=['tweet'], inplace=True)
    df = df.rename(columns={'tweet_processed': 'tweet'})

    print("Total tweets: ", df.shape)

    # split data into train(80%) and test (20%)
    x_train, x_test, y_train, y_test = train_test_split(df[['tweet_id', 'tweet', 'label', 'group']],
                                                        df['label'],
                                                        stratify=df['label'],
                                                        test_size=0.20)  # train-test split: 80%-20%

    # split train set further into train (65%) and val set (15%)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                      stratify=y_train,
                                                      test_size=15 / 80)  # train-val split: 65%-15%

    x_train.to_csv(f'../data/{data}/{data.lower()}_train.csv', sep='\t', index=False)
    x_val.to_csv(f'../data/{data}/{data.lower()}_valid.csv', sep='\t', index=False)
    x_test.to_csv(f'../data/{data}/{data.lower()}_test.csv', sep='\t', index=False)

    print("Completed!!")

if __name__ == '__main__':
    data = sys.argv[1] # Datasets: DWMW17, FDCL18, HatEval19
    input_file = sys.argv[2] # output obtained using binary_formatter (e.g., dwmw17_tweet_label_group_binary.csv)
    stratified_sample(data, input_file)