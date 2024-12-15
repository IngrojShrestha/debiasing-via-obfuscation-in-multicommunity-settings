import sys
import argparse
import Obfuscator as obf

print("Obfuscating FPs from validation set")

OO7_classifier = sys.argv[1]

# MLP as 007-classifier
if OO7_classifier == 'MLP':
    STATE = sys.argv[2]
    DATA = sys.argv[3]
    RUN = sys.argv[4]
    parser_ = obf.get_parser_()
    args = parser_.parse_args([STATE, DATA, RUN])
    obf.load_model_template(args)
    obf.main(
        conversion_file='/path/to/valid_fps.csv', # extracted false positive instances from validation set
        obfuscator_name='MLP',
        embedding_loc='/path/to/glove.6B.300d.txt')

# BERT-base as 007-classifier
elif OO7_classifier == 'BERT_base':
    obf.main(conversion_file='/path/to/valid_fps.csv', 
             obfuscator_name='BERT_base',
             embedding_loc='/path/to/glove.twitter.27B.100d.txt')
    
# MIDAS as 007-classifier
elif OO7_classifier == 'MIDAS':
    obf.main(
        conversion_file='/path/to/valid_fps.csv', 
        obfuscator_name='MIDAS',
        embedding_loc='/path/to/glove.twitter.27B.100d.txt')

# NULI as 007-classifier
elif OO7_classifier == 'NULI':
    obf.main(
        conversion_file='/path/to/valid_fps.csv',
        obfuscator_name='NULI',
        embedding_loc='/path/to/glove.twitter.27B.100d.txt')

# SentimentClassifier as 007-classifier
elif OO7_classifier == 'SC':
    obf.main(
        conversion_file='/path/to/valid_fps.csv',
        obfuscator_name='SentimentClassifier',
        embedding_loc='/path/to/glove.twitter.27B.100d.txt')
