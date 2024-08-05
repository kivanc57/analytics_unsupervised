from spacy import load

#OS Modules
from os.path import join
from os import listdir

#Load configs
from config.common_config import configure_logging

#Necessary modules for logging
import logging
from os.path import basename

#Configure logging
script_name = basename(__file__)[:-3]
configure_logging(script_name)
logger = logging.getLogger(__name__)

def initialize_nlp():
  try:
    nlp = load('en_core_web_sm')
    logger.info("Language model is succefully initialized")
    return nlp
  
  except Exception as e:
    logger.exception(f"Failed: {e} during initialization of language model")
