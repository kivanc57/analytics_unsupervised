import logging

#Numerical / Statistical modules
from pandas import DataFrame
from numpy import log, unique
from scipy.spatial.distance import pdist, squareform

#Visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from seaborn import scatterplot

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

def plot_dimensionality_reduction(df, n, technique, destination):
    try:
        if n == 1:
            plt.figure(figsize=(10, 5))
            plt.plot(df[df.columns[0]], c='blue', marker='o', markersize=5, linestyle='-')
            plt.title(f"1D Visualization of {technique}", fontweight = "bold")
            plt.xlabel(df.columns[0])
            plt.ylabel('Value')
        elif n == 2:
            plt.figure(figsize=(10, 7))
            scatterplot(data=df, x=df.columns[0], y=df.columns[1], c='blue', edgecolors='black', marker='o', s=50)
            plt.title(f"2D Visualization of {technique}", fontweight = "bold")
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
        elif n == 3:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[df.columns[0]], df[df.columns[1]], df[df.columns[2]], c='blue', edgecolors='black', marker='o', s=50)
            ax.set_title(f"3D Visualization of {technique}", fontweight = "bold")
            ax.set_xlabel(df.columns[0])
            ax.set_xlabel(df.columns[1])
            ax.set_zlabel(df.columns[2])
        else:
            raise ValueError("The 'n_components' must be between 1 and 3")

        plt.savefig(destination)
        plt.close()
        logger.info(f"Created {technique} graph for {n}D in {destination}")

    except Exception as e:
        logger.exception(f"{technique} graph failed: {e} for {n}D in {destination}")

def main():
    try:
        input_folder_name, output_folder_name = FOLDER_NAMES['input_folder_name'], FOLDER_NAMES['output_folder_name']
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
