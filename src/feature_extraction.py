#Numerical / Statistical modules
from pandas import DataFrame
from numpy import log

#Load configs
from config.common_config import configure_logging

#Necessary default modules for logging
import logging
from os.path import basename

#Configure logging
script_name = basename(__file__)[:-3]
configure_logging(script_name)
logger = logging.getLogger(__name__)

def get_bow_df(texts_dict, global_list):
    #Create vectors for each text name based on the length of global_list
    bows = {name: [0] * len(global_list) for name in texts_dict}

    try:
        for name, lemma_list in texts_dict.items():
            for lemma in lemma_list:
                if lemma in global_list:
	                vector_index = global_list.index(lemma) #Get the index of hit on global_list
	                bows[name][vector_index] += 1 #And increase by one on bow
        bow_df = DataFrame(bows).T
        bow_df.columns = global_list
        logger.info("Bow extraction succeeded")
        return bow_df
    
    except Exception as e:
        logger.exception(f"Bow extraction failed: {e}")
        return None

def apply_tfidf(df):
    try:
        N = df.shape[0] #N = number of documents
        tf = df.div(df.sum(axis=1), axis=0) #tf = Term Frequency (TF) 
        df_count = (df > 0).sum(axis=0) #df = Document Frequency (DF)
        idf = log((N + 1) / (df_count + 1) + 1) #idf = Inverse Document Calculation (IDF)
        tfidf = tf * idf
        logger.info("Tf-Idf applied")
        return tfidf
    
    except Exception as e:
        logger.exception(f"Tf-Idf failed: {e}")
        return None

def apply_z_score(df):
    try:
        mean = df.mean() #Get mean
        std_dev = df.std(ddof=1) #Get standard deviation
        z_scores = (df - mean) / std_dev #Get z-score
        logger.info("z-score applied")
        return z_scores
    
    except Exception as e:
        logger.exception(f"z-score filed: {e}")
        return None
