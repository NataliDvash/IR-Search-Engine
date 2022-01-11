# IR-Search-Engine
Information Retrieval 2021- Class final project

Created by Itay Aharony and Natali Dvash
___________________________________________
## 1. Search Frontend

The given frontend application

* *CosineSimilarity_body* - Calculate the Cosine Similarity of a query with all its relevant docs.
* *BM25_from_index* - Calculate the BM25 of a query with all its relevant docs.
* *heapq_nlargest* - Return sorted list of top n values
* *tokenizer* - Return list of tokens from string
* *score_title_replace* - Replace [(doc_id, score)...] with [(doc_id, title)...]
* *search_anchor* - Required function
* *search_title* - Required function
* *search_text* - Required function
* *search* - Required function
* *get_pagerank* - Required function
* *get_pageview* - Required function


## 2. Preprocessing

In this notebook we pre-processed the data for the Inverted Index as discussed in the report.
All the processing was created via map-reduce functions. the functions we used are:

* *tokenization* - Rokenize a string into a list of tokens
* *get_tfidf_norm* - Calculate the normalization factor for CosSim for each doc


## 3. Inverted Index

Most of the implementation in this file has been taken from assignment3, the function we added is:

* *read_a_posting_list* - Reads a specific posting list by term.


