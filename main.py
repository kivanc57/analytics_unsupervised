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

def main():
    try:
        input_folder_name, output_folder_name = FOLDER_NAMES['input_folder_name'], FOLDER_NAMES['output_folder_name']
        nlp = initialize_nlp()
        output_reduction_name = 'graph.png'
        source = get_join_path(input_folder_name, is_sample=True)
        reduction_destination = get_join_path(output_folder_name, output_reduction_name, is_sample=True)

        texts_dict, global_list = get_texts_dict_global_dict(nlp, source)
        bow_df = get_bow_df(texts_dict, global_list)
        #emphasized_bow = apply_tfidf(bow_df)
        #normalized_bow = apply_z_score(emphasized_bow)
        df, n, technique = apply_technique(bow_df, n_components=2, metric='cosine',
                                          compute_distance=True, technique='TSNE')
        plot_dimensionality_reduction(df, n, technique, reduction_destination)
        logger.info(f"Successfully executed: {basename(__file__)}")

    except Exception as e:
        logger.exception(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
