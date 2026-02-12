import sys, csv, json
csv.field_size_limit(sys.maxsize)
import math
import numpy as np
import pandas as pd

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

        for word in preprocess(email) :
            if word not in index :
                index[word] = {}
            if docid not in index[word]:
                index[word][docid] = 0
            index[word][docid] += 1

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

def BIM(corpus, query, df):
    weights = calculate_weights(df, len(corpus))
    scored_documents = []
    preprocessed_query = preprocess(query)
    doc_id = 0
    for document in corpus:
        scored_documents.append((document, score_document(document, preprocessed_query, weights), doc_id))
        doc_id += 1
    ranked_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
    for document in ranked_documents[:10]:
        print(f"Document score: {document[1]} \nDocument email message preview: {document[0][:150]}")
    return ranked_documents


def calculate_doc_length(document):
    return len(preprocess(document))

def average_length_of_docs(corpus):
    sum = 0
    for doc in corpus:
        sum += calculate_doc_length(doc)
    return sum/len(corpus)

def term_frequency(term, doc_id, index):
    if str(doc_id) in index[term]:
        return index[term][str(doc_id)]
    return 0

def RSV_d(doc_id, df, N, k_1, b, L_d, L_ave, query, index):
    sum = 0
    for term in preprocess(query):
        tf = term_frequency(term, doc_id, index)
        sum += math.log(N/df[term])*(((k_1+1)*tf)/(k_1*((1-b)+b*(L_d/L_ave))+tf))
    return sum


def Okapi_BM25(corpus, index, query, k_1, b, df):
    L_ave = average_length_of_docs(corpus)
    doc_id = 0
    scored_documents = []
    for document in corpus:
        L_d = calculate_doc_length(document)
        score = RSV_d(doc_id, df, len(corpus), k_1, b, L_d, L_ave, query, index)
        scored_documents.append((document, score, doc_id))
        doc_id += 1
    ranked_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
    for document in ranked_documents[:10]:
        print(f"Document score: {document[1]} \nDocument email message preview: {document[0][:150]}")
    return ranked_documents

def calculate_tf_idf_scores(index, query, N, df):
    tf_idf_scores = {}
    tf_idf_norm = {}
    tf_idf_norm["query"] = 0
    for term in set(preprocess(query)):
        if term not in index:
            continue

        idf = math.log(N/df[term])

        tf_query = query.count(term)
        tf_idf_norm["query"] += (tf_query*idf)**2

        for doc_id, tf in index[term].items():
            print(doc_id, tf, term)
            if doc_id not in tf_idf_scores:
                tf_idf_scores[doc_id] = 0
                tf_idf_norm[doc_id] = 0
            tf_idf_scores[doc_id] += tf*idf*tf_query*idf
            tf_idf_norm[doc_id] += (tf*idf)**2

    print(tf_idf_scores)
    print(tf_idf_norm)
    return tf_idf_scores, tf_idf_norm

def sim(scores, norms, doc_id):
    if doc_id not in scores:
        return 0
    return scores[doc_id]/(math.sqrt(norms[doc_id])*math.sqrt(norms["query"]))


def vector_space_model(corpus, index, query):
    doc_id = 0
    scored_documents = []
    scores, norms = calculate_tf_idf_scores(index, query, len(corpus), df)
    for document in corpus:
        score = sim(scores, norms, str(doc_id))
        scored_documents.append((document, score, doc_id))
        doc_id += 1
    ranked_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)
    for document in ranked_documents[:10]:
        print(f"Document score: {document[1]} \nDocument email message preview: {document[0][:150]}")
    return ranked_documents

def make_table(documents):
    data = []
    for document in documents[:10]:
        data.append([document[2], document[1], calculate_doc_length(document[0])])
    print_table(data)

def print_table(data):
    table = pd.DataFrame(data, columns=["Doc_id", "Score", "Length"])
    table.index = np.arange(1, 11)
    print(table)

#corpus, index = build_index(limit=1000) # build index for small subset
#corpus, index = build_index() # build full index

#write_to_disk(corpus, 'enron_text2.json')
#write_to_disk(index,  'enron_index2.json')

#print("#documents = {}, #terms = {}".format(len(corpus), len(index)))

corpus2 = read_from_disk('enron_text2.json')
index2  = read_from_disk('enron_index2.json')

query = "forecast" #"traveling is fun if forecast is great for business" #"here is our forecast"
df = calculate_df(index2)

print("#documents = {}, #terms = {}".format(len(corpus2), len(index2)))

print("BIM document scores:")
bim = BIM(corpus2, query, df)
print("Okapi BM25 document scores:")
bm25 = Okapi_BM25(corpus2, index2, query, 1.2, 0.75, df)
print("Vector space model document scores:")
vsm = vector_space_model(corpus2, index2, query)
print("BIM: ")
make_table(bim)
print("VSM: ")
make_table(vsm)
print("BM25: ")
make_table(bm25)