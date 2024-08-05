from sklearn.preprocessing import StandardScaler

#Dimensionality Reduction modules
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import MDS, TSNE
#Clustering modules
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster

#Numerical / Statistical modules
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform

#Necessary modules for logging
import logging
from os.path import basename

#Load configs
from config.common_config import get_join_path, configure_logging
from config.constants import FOLDER_NAMES

#Configure logging
script_name = basename(__file__)[:-3]
configure_logging(script_name)
logger = logging.getLogger(__name__)

def apply_metric(df, metric='cosine'):
    try:
        distance_pairwise = pdist(df, metric=metric)
        distance_matrix = DataFrame(squareform(distance_pairwise), index=df.index, columns=df.index)
        logging.info(f"Metric: {metric} applied")
        return distance_matrix

    except Exception as e:
        logging.exception(f"Metric: {metric} failed: {e}")
        return None

def apply_technique(df, n_components=2, technique='PCA',
                    metric='cosine', compute_distance=True, **kwargs):
    cluster_labels = None
    clustering_list = ['KMEANS', 'DBSCAN', 'HC']
    reduction_list = ['PCA', 'SVD', 'TRUNCATEDSVD', 'MDS', 'TSNE']
    technique = technique.upper()

    if n_components not in range(1, 4):
        raise ValueError("The 'n_components' must be between 1 and 3")

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
        
        n = result_df['Cluster'].values_counts() if cluster_labels is not None else n_components 
        return result_df, n, technique

    except Exception as e:
        logger.exception(f"Failed applying technique: {e}")
        return None, None, None
