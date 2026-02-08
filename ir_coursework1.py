import sys, csv, json
csv.field_size_limit(sys.maxsize)
import math

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# 
# 1. download enron emails from here + unzip:
#   https://www.kaggle.com/datasets/acsariyildiz/the-enron-email-dataset-parsed
#   $ unzip archive.zip
#
# 2. setup a virtual environment and install dependencies
#   $ python3 -m venv .env
#   $ . ./.env/bin/activate
#   $ pip install nltk
#
# 3. run the following in the Python REPL before running this script:
#   >>> import nltk
#   >>> nltk.download('stopwords')
#
# 4. run script to build indices
#   $ python ir_coursework1.py
# 

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def parse_enron() :
    with open('enron_mails.csv', newline='\n') as f :
        reader = csv.reader(f) 
        next(reader)

        for row in reader :
            yield row[17]

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def remove_stopwords(s) : 
    return [ w for w in s if w not in stop_words ]

def remove_numbers_punctuation(s) :
    return [ w for w in s if w.isalpha() ]

def stemming(s) :
    return [ stemmer.stem(w) for w in s ]

def preprocess(s) :
    tokens = [ w for w in word_tokenize(s.lower()) ]
    tokens = remove_numbers_punctuation(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return tokens

def build_index(limit=None) :
    corpus = []
    index = {}
    
    for email in parse_enron() :
        corpus.append(email)
        docid = len(corpus) - 1
        print('\rreading email #{}...'.format(docid), end='', file=sys.stderr)

        for word in set(preprocess(email)) :
            if word not in index :
                index[word] = []
            index[word].append(docid)

        if limit and len(corpus) == limit :
            break

    print('\rreading email done!' + ' '*10)
    return corpus, index

def write_to_disk(obj, fname) :
    with open(fname, 'w') as f :
        json.dump(obj, f)

def read_from_disk(fname) :
    with open(fname) as f :
        return json.load(f)


# solutions

def calculate_df(index):
    df = {word: len(documents) for word, documents in index.items()}
    return df

def BIM_weight(df, N):
    return math.log((N - df + 0.5)/(df + 0.5))

def calculate_weights(df, N):
    weights = {}
    for term, freq in df.items():
        weights[term] = BIM_weight(freq, N)
    return weights

def score_document(document, query, weights):
    return sum(weights[term] for term in query if term in document)

def BIM(corpus, index, query):
    df = calculate_df(index)
    weights = calculate_weights(df, len(corpus))
    scored_documents = []
    preprocessed_query = preprocess(query)
    for document in corpus:
        scored_documents.append((document, score_document(document, preprocessed_query, weights)))
    ranked_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
    for document in ranked_documents[:10]:
        print(f"Document score: {document[1]} \nDocument email message preview: {document[0][:150]}")

#corpus, index = build_index(limit=1000) # build index for small subset
#corpus, index = build_index() # build full index

#write_to_disk(corpus, 'enron_text.json')
#write_to_disk(index,  'enron_index.json')

#print("#documents = {}, #terms = {}".format(len(corpus), len(index)))

corpus2 = read_from_disk('enron_text.json')
index2  = read_from_disk('enron_index.json')

print("#documents = {}, #terms = {}".format(len(corpus2), len(index2)))

BIM(corpus2, index2, "job Monday trip")