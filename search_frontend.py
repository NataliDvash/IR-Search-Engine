from flask import Flask, request, jsonify
from inverted_index_gcp import *
from super_inverted_index_gcp import *
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import math
import os
from contextlib import closing
import math
from itertools import chain
import time
import heapq
import json

############# Evaluation methods  #############

#Cosine similarity for the body search function
def CosineSimilarity_body(index, query):
    ''' Parameters:
      -----------
      index - InvertedIndex instance of body
      query - list of tokens representing the query
      Returns:
      -----------
      CosSim_per_doc - dict in the form of {doc_id: CosSim, ...}
    '''
    len_q = len(query)
    query = Counter(query).items()  # Count the tokens in the query
    
    # Calc tf-idf for query
    query = [(item[0], (item[1]/len_q) * np.log10(len(index.doc_len) / index.df[item[0]])) for item in query if item[0] in index.df] 
    
    dot_product = {}  # {word: {doc1:tfidf doc1 dot query, doc2:tfidf doc1 dot query,...}, word2:....}

    # Calc and tf-idf dot product of doc and query
    for item in query:
        token, q_tfidf = item
        pl = index.read_posting_list(token) # Get the posting list of the token
        for doc_id, freq in pl:
            try:
                tf = freq / index.doc_len[doc_id]
                idf = np.log10(len(index.doc_len) / index.df[token])
                # find the dot product
                doc_dot_query = tf * idf * q_tfidf
                # sum all the results per doc id
                dot_product[doc_id] = dot_product.get(doc_id, 0) + doc_dot_query
            except:
                dot_product[doc_id] = 0
    # Calc tf-idf for query - didn't do it, ask ET if relavent [pineapple]
    cos_dic = {}
    # Query normalizing factor for cosine similarity
    q_norm = math.sqrt(sum( [tup[1] ** 2 for tup in query] ) )
    
    # Normalize the dot product
    for doc_id in dot_product.keys():
        try:
            cos_dic[doc_id] = dot_product[doc_id] / (q_norm * index.doc_nf[doc_id])
        except:
            cos_dic[doc_id] = 0
    
    return cos_dic


# BM25 using our own index implemantation = super_index
class BM25_from_index():
    def __init__(self, index, idf, AVGDL, k1=1.5, b=0.75):
            self.b = b
            self.k1 = k1
            self.index = index
            self.idf = idf
            self.N = len(index.doc_len)
            self.AVGDL = AVGDL
                    
    def search(self, tokens):
        scores = {}
        for term in tokens:
            for doc_id, freq in self.index.read_posting_list(term):
                try:
                    idf = self.idf[term]
                    numerator = idf * freq * (self.k1+1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * (self.index.doc_len[doc_id] / self.AVGDL))
                    score = idf * (numerator / denominator)
                    scores[doc_id] = scores.get(doc_id,0) + score
                except:
                    continue
        return scores

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):

        # Reading the {id:pageViews} dic
        with open("postings_gcp/pageviews-202108-user.pkl", 'rb') as f:
            self.wid2pv = pickle.loads(f.read())
        # Creating normalization factor
        self.wid2pv["nf"] = max(self.wid2pv.values())

        # Reading the {id:title} dic
        with open("postings_gcp/id_title.pkl", 'rb') as f:
            self.doc_title_pairs = pickle.loads(f.read())
        
        # Reading the {id:pagerank} dic
        with open("postings_gcp/id_pagerank.pkl", 'rb') as f:
            self.id_pagerank_pairs = pickle.loads(f.read())

        # Creating normalization factor
        self.id_pagerank_pairs["nf"] = max(self.id_pagerank_pairs.values())

        # Reading title_index
        with open('postings_gcp/title_index.pkl', 'rb') as f:
            self.title_index = pickle.load(f)

        # Reading text_index
        with open('postings_gcp/text_index.pkl', 'rb') as f:
            self.text_index = pickle.load(f)

        # Reading anchor_index
        with open('postings_gcp/anchor_index.pkl', 'rb') as f:
            self.anchor_index = pickle.load(f)

        # Idf for the BM25
        with open('postings_gcp/idf.pkl', 'rb') as f:
            self.idf = pickle.load(f)
    
        # Avg doc len for the BM25
        self.AVGDL = sum(self.text_index.doc_len.values())/len(self.text_index.doc_len)

        # Init tokenaizer
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


############# Helpers #############

# Using heapq k largest for getting result quicker. 
def heapq_nlargest(unsorted, k):
    return sorted(heapq.nlargest(k, unsorted, key=lambda x: x[1]), key=lambda x: x[1], reverse=True)


# Tokenizer function
def tokenizer(query):
    return [token.group() for token in app.RE_WORD.finditer(query.lower())]


# Replace the score with title like requaried
def score_title_replace(lst):
    new_lst = []
    for x in lst:
        new_lst.append((x[0], app.doc_title_pairs.get(x[0], "no title")))
    return new_lst




############# Supported functions #############

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    # BEGIN SOLUTION

    # Prepare the query
    tokens = tokenizer(query)


    ### Find CoSim_scores ###

    # Calculate [(id,CosSim_score),...] 
    cossim_list = CosineSimilarity_body(app.text_index, tokens).items()

    # Get top 100 docs by CosSim
    CoSim_scores = heapq.nlargest(100, cossim_list, key=lambda x: x[1])


    ### Find BM25_scores ###

    # Init the BM_class
    BM25 = BM25_from_index(app.text_index, app.idf, app.AVGDL, 1.5, 0.4)

    # Calculate [(id,BM25_score),...]
    bm25_list = BM25.search(tokens).items()

    # Get top 100 docs by BM25
    BM25_scores = heapq.nlargest(100, bm25_list, key=lambda x: x[1])


    ### Merge CosSim & BM25 scores ###
    
    score = {}
    
    for tup in BM25_scores:
        score[tup[0]] = score.get(tup[0], 0) + tup[1] * 0.7
    
    for tup in CoSim_scores:
        score[tup[0]] = score.get(tup[0], 0) + tup[1] * 0.3
    

    ### Select top 100 using PRank and PView ###

    # Take into consideration Prank and Pview(normalized), and select top 100 in descending order
    top_100 = sorted([(x[0], x[1] * ( 0.5 * app.id_pagerank_pairs.get(x[0], 0) / app.id_pagerank_pairs["nf"] + 0.5 * app.wid2pv.get(x[0], 0) / app.wid2pv["nf"])) for x in score.items()], key=lambda x:x[1], reverse=True)


    # Get the [(id,title)..] of relevant id's
    res = score_title_replace(top_100)

    # END SOLUTION
    return jsonify(res)
    


@app.route("/search_body")
def search_body():  # delete the query arg in real version! [pineapple]
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')

    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # Prepare the query
    tokens = tokenizer(query)

    # Calculate [(id,CoSim_score),...] 
    cossim_list = CosineSimilarity_body(app.text_index, tokens).items()

    # Sort the above
    top_100_CosSim = heapq_nlargest(cossim_list, 100)

    # Get the [(id,title)..] of relevant id's
    res = score_title_replace(top_100_CosSim)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')

    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # Prepare the query
    tokens = tokenizer(query)

    # Init Counter
    _count = Counter()

    # Counting the NUMBER OF QUERY WORDS that appear in the title
    for t in tokens:
        for x in app.title_index.read_posting_list(t):
            if x[0]==0:
                continue
            _count.update([x[0]])
    
    # Order the above in descending order
    binary_sorted = _count.most_common()

    # Get the [(id,title)..] of relevant id's
    res = score_title_replace(binary_sorted)

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')

    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # Prepare the query
    tokens = tokenizer(query)

    # Init Counter
    _count = Counter()

    # Counting the NUMBER OF QUERY WORDS that appear in the title
    for t in tokens:
        for x in app.anchor_index.read_posting_list(t):
            if x[0]==0:
                continue
            _count.update([x[0]])
    
    # Order the above in descending order
    binary_sorted = _count.most_common()

    # Get the [(id,title)..] of relevant id's
    res = score_title_replace(binary_sorted)


    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # List of PageRank scores that correrspond to the provided article IDs.
    for key in wiki_ids:
        res.append(app.id_pagerank_pairs.get(key, None))
        
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # List of page view numbers from August 2021 that correrspond to the provided list article IDs.
    for key in wiki_ids:
        res.append(app.wid2pv.get(key, None))

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
