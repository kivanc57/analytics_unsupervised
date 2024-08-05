#Visualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from seaborn import scatterplot

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
