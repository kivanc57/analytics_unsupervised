from config.constants import FOLDER_NAMES
from config.nlp_config import initialize_nlp
from config.common_config import get_join_path, configure_logging
from src.preprocessing import get_texts_dict_global_dict
from src.feature_extraction import get_bow_df, apply_tfidf, apply_z_score
from src.dimensionality_reduction import apply_technique
from src.visualization import plot_dimensionality_reduction

#Necessary modules for logging
import logging
from os.path import basename

#Configure logging
script_name = basename(__file__)[:-3]
configure_logging(script_name)
logger = logging.getLogger(__name__)
logger.info("Package is initialized")

#Metadata
__version__ = "1.0.0"
__date__ = "05-08-2024"
__email__ = "kivancgordu@hotmail.com"
__status__ = "production"

__all__ = [
  'FOLDER_NAMES', 'initialize_nlp', 'get_join_path', 'configure_logging',
  'get_texts_dict_global_dict', 'get_bow_df', 'apply_tfidf', 'apply_z_score',
  'apply_technique', 'plot_dimensionality_reduction'
]
