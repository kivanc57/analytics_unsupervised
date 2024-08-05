#OS Modules
from os.path import join
from os import listdir

#Load language modal
from spacy.lang.en.stop_words import STOP_WORDS

#Load configs
from config.common_config import configure_logging

#Necessary modules for logging
import logging
from os.path import basename

#Configure logging
script_name = basename(__file__)[:-3]
configure_logging(script_name)
logger = logging.getLogger(__name__)

def extract_lemmas(text):
    lemma_list = list()

    try:
        #Add its lemma if not in stop_words or is alphabetical
        for token in text:
            if token not in STOP_WORDS and token.is_alpha:
                lemma_list.append(token.lemma_)
        logger.info(f"Lemma list extraction succeeded")
        return lemma_list
    
    except Exception as e:
        logger.exception(f"Lemma list extraction failed: {e}")
        return None

def get_texts_dict_global_dict(nlp, source):
    texts_dict = dict()
    lemma_list = list()
    global_list = list()

    try:
        for file_name in listdir(source):
            if file_name.endswith('.txt'):
                logger.info(f"Reading file: {file_name}")
                file_path = join(source, file_name)
                try:
                    with open(file_path, mode='r', encoding='utf-8', errors='replace') as f:
                        doc = nlp(f.read())
                        lemma_list = extract_lemmas(doc)
                        if lemma_list is None:
                            return None, None
                        global_list += lemma_list
                        texts_dict[file_name] = lemma_list
                    logger.info(f"Reading succeeded in {source}")
                
                except Exception as e:
                    logger.exception(f"Reading failed: {e} in {source}")
                    return None, None

        logger.info(f"Extraction succeeded in {source}")
        return texts_dict, global_list

    except Exception as e:
        logger.exception(f"Extraction failed: {e} in {source}")
        return None, None
