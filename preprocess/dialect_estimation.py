import numpy as np
import pandas as pd
import sys
import re

# https://github.com/slanglab/twitteraae
if '../twitteraae/code/' not in sys.path:
    sys.path.insert(0, '../twitteraae/code/')

import predict

def load_raw_data(DATA, PATH_TO_RAW_DATA):
    '''
    Returns dataframe with only tweet and class
    '''
    
    if DATA == 'FDCL18' or DATA == 'HatEval19':
        df = pd.read_csv(PATH_TO_RAW_DATA, sep='\t')
    elif DATA=='DWMW17':
        df = pd.read_csv(PATH_TO_RAW_DATA, sep=',')
    else:
        print("provide valid dataset")
        sys.exit(0)

    df = df[['tweet', 'class']]
    return df


def parse_text(text):
    # remove new line
    text = text.replace("\n", '')

    # remove links
    text = re.sub(r"http\S+", "", text)

    # remove hash
    text = re.sub(r'#\w+ ?', '', text)

    # remove mentions
    text = re.sub(r"@\S+", "", text)

    # remove extra white spaces
    text = ' '.join(text.split())

    # remove RT
    text = text.replace('RT', '')

    # remove leading and trailing white space
    text = text.strip()

    # remove double quote
    text = text.replace('"', '')

    return text

def extract_group(df):
    df_grp = pd.DataFrame(columns=['tweet_processed', 'tweet_id', 'AE', 'Hispanic', 'Other' , 'White',
                                   'group'])  # "tweet_id is generated randomly for reference"

    predict.load_model()
    none_count = 0

    for index, row in df.iterrows():
        # print(index)
        tweet = row['tweet']
        tweet_processed = parse_text(tweet)
        dst = predict.predict(tweet_processed.split())
        group = np.argmax(dst)

        if dst is None:
            none_count += 1
            
            temp_dict = {
                    'tweet_processed': [tweet_processed],
                    'tweet_id': [index + 1],
                    'AE': [-1],
                    'Hispanic': [-1],
                    'Other': [-1],
                    'White': [-1],
                    'group': [-1]
                }
        else:
            temp_dict = {
                    'tweet_processed': [tweet_processed],
                    'tweet_id': [index + 1],
                    'AE': [dst[0]],
                    'Hispanic': [dst[1]],
                    'Other': [dst[2]],
                    'White': [dst[3]],
                    'group': [group]
                 }
        temp_df = pd.DataFrame(temp_dict)
        df_grp = pd.concat([df_grp, temp_df], ignore_index=True)    
        
    print("None Dialect: ", none_count)
    return df_grp


def merge_tweet_group(df, df_grp):
    df_all = pd.concat([df, df_grp], axis=1, ignore_index=True)
    df_all.columns = ['tweet', 'label', 'tweet_processed', 'tweet_id', 'AE', 'Hispanic', 'Asian', 'White', 'group']
    df_all['tweet'] = df_all['tweet'].str.replace("\n", '')
    df_all['tweet'] = df_all['tweet'].str.strip('"')  # remove double quote from start and end of a tweet text
    df_all['tweet'] = df_all['tweet'].str.strip()  # remove leading and trailing white space
    return df_all

def preprocess(DATA, PATH_TO_RAW_DATA, output_filename):
        
    # load raw data
    df_raw = load_raw_data(DATA, PATH_TO_RAW_DATA)  # contains tweet_text and label

    # obtain group for corresponding tweeet
    df_grp = extract_group(df_raw)

    # combine tweet_text, label and group
    df_all = merge_tweet_group(df_raw, df_grp)

    # save df_all
    df_all.to_csv(output_filename, sep='\t', index=False)

if __name__ == "__main__":
    DATA = sys.argv[1].strip()
    PATH_TO_RAW_DATA = sys.argv[2].strip() # original dataset file
    output_filename = sys.argv[3].strip() # dwmw17_tweet_label_group.csv, fdcl18_tweet_label_group.csv, hateval19_tweet_label_group.csv
    
    preprocess(DATA, PATH_TO_RAW_DATA, output_filename)