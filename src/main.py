#OS modules
from os.path import join, basename
from os import listdir
import logging

#Numerical / Statistical modules
from spacy import load
from spacy.lang.en.stop_words import STOP_WORDS
from pandas import DataFrame
from numpy import log, unique
from scipy.spatial.distance import pdist, squareform

#Visualization
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import silhouette_samples
from scipy.cluster.hierarchy import dendrogram

#Preprocessing
from sklearn.preprocessing import StandardScaler
#Dimensionality Reduction modules
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE
#Clustering modules
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster

#Load configs
from config.common_config import get_join_path, configure_logging
from config.constants import FOLDER_NAMES

#Load language modal
nlp = load('en_core_web_sm')

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

def apply_metric(df, metric='cosine'):
    try:
        distance_pairwise = pdist(df, metric=metric)
        distance_matrix = DataFrame(squareform(distance_pairwise), index=df.index, columns=df.index)
        logging.info(f"Metric: {metric} applied")
        return distance_matrix

    except Exception as e:
        logging.exception(f"Metric: {metric} failed: {e}")
        return None

def apply_technique(df, n_components=3, technique='PCA',
                    metric='cosine',compute_distance=True, **kwargs):
    cluster_labels = None
    
    technique = technique.upper()
    clustering_list = ['KMEANS', 'DBSCAN', 'HC']
    reduction_list = ['PCA', 'SVD', 'TRUNCATEDSVD', 'MDS', 'TSNE']

    try:
        if technique in clustering_list or technique in reduction_list: #Check if the the technique is available

        #Check if a metric is given to adjust distance to be used for following models
        #DBSCAN using its own during clustering so we exempt this case
            if compute_distance and technique != 'DBSCAN':
                df = apply_metric(df, metric=metric)
                if df is None:
                    return None

            #Standardize the data if it's a clustering technique
            #Following, apply the technique
            if technique in clustering_list:
                df = StandardScaler().fit_transform(df)
                if technique == 'KMEANS':
                    n_clusters = kwargs.get('n_clusters', 3) #Default  n_clusters= 3
                    model = KMeans(n_clusters=n_clusters)
                    cluster_labels = model.fit_predict(df)

                if technique == 'DBSCAN':
                    eps = kwargs.get('eps', 0.5) # Default eps=0.5
                    min_samples = kwargs.get('min_samples', 5) #Default min_samples=5
                    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                    cluster_labels = model.fit_predict(df)

                if technique == 'HC':
                    method = kwargs.get('method', 'ward') #Default method=ward
                    t = kwargs.get('t', 1.0) #Default threshold=1.0
                    linkage_matrix = linkage(df, method=method)
                    cluster_labels = fcluster(linkage_matrix, t=t, criterion='distance')

                result_df = DataFrame(data=cluster_labels, columns=['Cluster'])


            elif technique in reduction_list:
                if technique == 'PCA':
                    model = PCA(n_components=n_components)
                if technique in ['SVD', 'TRUNCATEDSVD']:
                    model = TruncatedSVD(n_components=n_components)
                if technique == 'MDS':
                    model = MDS(n_components=n_components)
                if technique == 'TSNE':
                    #Ensure perplexity is valid
                    perplexity = kwargs.get('perplexity', 2)
                    if perplexity >= df.shape[0]:
                        raise ValueError("Perplexity must be less than the number of documents")
                    model = TSNE(n_components=n_components, perplexity=perplexity)

                model_components = model.fit_transform(df)
                result_df = DataFrame(data=model_components, columns=[f'{technique}_{i+1}' for i in range(n_components)])

            else:
                raise ValueError (f"Unknown technique: {technique}. Please select between 'PCA', '(Truncated-)SVD', 'MDS', 'tSNE', 'kMeans, DBSCAN, or 'HC'")
        
        if cluster_labels:
            n = result_df['Cluster'].values_counts()
        else:
            n = n_components
        return result_df, n

    except Exception as e:
        logger.exception(f"Failed applying technique: {e}")
        return None

# def plot_dimensionality_reduction(destination, df, n, title='Dimensionality Reduction Plot'):
#     if n == 2:
#         plt.figure(figsize=(10, 7))
#         plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c='blue', edgecolors='black', s=50)
#         plt.xlabel('Component 1')
#         plt.ylabel('Component 2')
#     if n == 3:
#         fig = plt.figure(figsize=(12, 9))
#         ax = plt.axes(projection='3d')
#         ax.scatter(df.iloc[:, 0], df[:, 1], df[:, 2], c='blue', edgecolors='black', s=50)
#         ax.set_xlabel('Component 1')
#         ax.set_xlabel('Component 2')
#         ax.set_zlabel('Component 3')
#     if title:
#         plt.title(title)

#     plt.savefig(destination)

# def plot_clusters(df, title='Cluster Plot'):
#     df.fit
#     labels = df.labels_
#     plt.figure(figsize=(10, 7))
#     unique_labels = unique(unique(labels))
#     colors = plt.cm.get_cmap('Dark2', len(unique_labels))

#     for k in unique_labels:
#         cluster_df = df[labels == k]
#         plt.scatter(cluster_df[df.iloc[:, 0]], cluster_df[df.iloc[:, 1]],
#                     s=50, c=[colors(k)], label=f'Cluster {k}')
#     plt.title(title)
#     plt.legend()
#     plt.savefig('/home/kgordu/cluster.png')

# def plot_dendrogram(df, title='Dendrogram'):
#     plt.figure(figsize=(10, 7))
#     dendrogram(df[:, ])
#     plt.title(title)
#     plt.savefig('/home/kgordu/dendrogram.png')

#def check_plot(technique, n) -- use to decide which plot after applying the technique
def main():
    try:
        input_folder_name, output_folder_name = FOLDER_NAMES['input_folder_name'], FOLDER_NAMES['output_folder_name']
        source = get_join_path(input_folder_name, is_sample=True)

        output_reduction_name = 'reduction_graph.png'
        reduction_destination = get_join_path(output_reduction_name, is_sample=True)

        texts_dict, global_list = get_texts_dict_global_dict(nlp, source)
        bow_df = get_bow_df(texts_dict, global_list)
        emphasized_bow = apply_tfidf(bow_df)
        normalized_bow = (emphasized_bow)
        print(normalized_bow)
        reducted_bow, n = apply_technique(bow_df, n_components=3, metric='cosine',
                                          compute_distance=True, technique='TSNE')
        print(reducted_bow)
        plot_clusters(reducted_bow)
        #plot_dimensionality_reduction(reduction_destination, reducted_bow, n, title='Dimensionality Reduction Plot')
        logger.info(f"Successfully executed: {basename(__file__)}")

    except Exception as e:
        logger.exception(f"Error in main function: {e}")

if __name__ == "__main__":
    main()
