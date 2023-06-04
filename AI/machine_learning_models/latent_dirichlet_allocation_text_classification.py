from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess


"""
Latent Dirichlet Allocation (LDA) is a generative probabilistic model used 
primarily in natural language processing for the task of topic modeling. 
It's designed to automatically discover the topics that a set of documents cover.

In LDA, each document is assumed to be a mixture of a small number of topics. 
The 'topics' are represented as a distribution over a fixed vocabulary. These 
topics are assumed to be 'latent' attributes, meaning they are not observed but 
rather inferred from the dataset_sites.

The basic idea is that documents are represented as random mixtures over latent
 topics, where each topic is characterized by a distribution of words. 
 The distributions are assumed to follow a Dirichlet distribution.
"""

class LDAModel:
    def __init__(self, documents, num_topics=10, random_state=100,
                 update_every=1, chunksize=100, passes=10, alpha='auto',
                 per_word_topics=True):
        self.documents = documents
        self.texts = [simple_preprocess(doc) for doc in self.documents]
        self.id2word = corpora.Dictionary(self.texts)
        self.corpus = [self.id2word.doc2bow(text) for text in self.texts]
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.id2word,
                                  num_topics=num_topics,
                                  random_state=random_state,
                                  update_every=update_every,
                                  chunksize=chunksize, passes=passes,
                                  alpha=alpha, per_word_topics=per_word_topics)

        """
        corpus: This is the collection of "bag of words" that represents your 
        documents. It's essentially a list of lists, where each inner list 
        represents a document and contains tuples of the form (word_id, 
        word_frequency).

        id2word: This is a gensim.corpora.Dictionary object mapping from word
         IDs to words. It's used to determine the vocabulary size, as well as 
         for debugging and topic printing.
        
        num_topics: This is the number of topics you want the LDA algorithm to 
        find. It's equivalent to the "K" in K-means. It's a hyperparameter you 
        have to set before training the model, and it can have a big impact
         on the results.
        
        random_state: This is the random seed. It's used for reproducibility 
        reasons, so you can get the same results if you run the script 
        multiple times.
        
        update_every: This parameter is specific to Gensim's implementation of 
        LDA. It determines how often the model parameters should be updated. 
        If set to 0, the model parameters are updated once at the end of each 
        pass. If set to a positive integer, say k, the model parameters are 
        updated every k documents.
        
        chunksize: This controls how many documents are processed at a time in 
        the training algorithm. Increasing chunksize will speed up training, 
        but it may also negatively affect the quality of the model.
        
        passes: The number of laps the model will take through corpus. The 
        greater the number of passes, the more accurate the model will be. A 
        lot of passes can be slow on a very large corpus.
        
        alpha: This parameter represents the per-document topic distribution. 
        High alpha indicates that each document is likely to contain a mixture 
        of most of the topics, and not any single topic specifically. 
        A low alpha indicates that a document is more likely to contain a 
        mixture of just a few topics. If you set alpha to 'auto', Gensim 
        will learn the alpha parameter from the dataset_sites.
        
        per_word_topics: If set to True, the model also computes a list of 
        topics in order of importance for each word, which can be accessed 
        using lda_model_loaded.get_document_topics(bow).
        """

    def save_model(self, filename):
        self.lda_model.save(filename)

    def load_model(self, filename):
        self.lda_model = LdaModel.load(filename)

    def predict_topic(self, document):
        new_doc = simple_preprocess(document)
        new_doc_bow = self.id2word.doc2bow(new_doc)
        doc_topics, word_topics, phi_values = self.lda_model.get_document_topics(
            new_doc_bow, per_word_topics=True)
        return doc_topics


# Usage
documents = ["Text Data.", "Text Data.", "Text Data.", "Text Data.",
             "Text Data.", "Text Data."]
lda = LDAModel(documents)

lda.save_model('lda_model.model')

lda.load_model('lda_model.model')

new_doc = "This is a new document."
topics = lda.predict_topic(new_doc)  # Predict topics for a new document
print(topics)
