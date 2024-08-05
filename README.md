# Analytics Unsupervised

## Overview ⋆｡‧˚ʚ🍓ɞ˚‧｡⋆

The **Analytics Unsupervised** project offers a comprehensive suite of Python modules designed for advanced text data processing, feature extraction, and visualization.

This package provides a robust framework for analyzing and transforming text data, applying dimensionality reduction techniques, and generating insightful visualizations.
Leveraging state-of-the-art technologies in mechine learning, it supports a wide range of tasks from preprocessing to visual representation.


👨🏻‍💻 **Advanced Data Processing Techniques** 👨🏻‍💻 -> Methods include `Bag of Words (BoW)`, `TF-IDF`, and `z-score` normalization to standardize features before applying dimensionality reduction techniques.

💥 **Dimensionality Reduction Methods** 💥 -> Techniques covered are `Principal Component Analysis (PCA)`, `Singular Value Decomposition (SVD)`, `Multi-Dimensional Scaling (MDS)`, and `t-Distributed Stochastic Neighbor Embedding (t-SNE)`.

🚨 **Clustering Algorithms** 🚨 -> Supported algorithms include `k-Means`, `DBSCAN`, and `Hierarchical Clustering (HC)` for grouping similar data points.

📶 **Visualization Options** 📶 -> Provides `1D`, `2D`, and `3D` visualization options using state-of-the-art libraries like `scikit-learn` and `matplotlib`, enabling insightful visualizations of data to uncover hidden patterns and insights.

## Table of Contents
1. [Introduction](#introduction)
2. [Core Components](#core_components)
3. [Project Structure](#project_structure)
4. [Visualizations](#visualizations)
  * [1D Result](#graph_1)
  * [2D Result](#graph_2)
  * [3D Result](#graph_3)
5. [Usage](#usage)
6. [License](#license)
7. [Contact](#contact)

---

## 1. Introduction <a name="introduction"></a>

This repository features a collection of Python scripts meticulously crafted for text data analysis and visualization.
Each script specializes in a distinct type of analysis or visualization, offering users a range of methods to derive valuable insights from text data.
By utilizing these scripts, users can explore and understand their text data through various analytical and visual techniques, making it easier to uncover meaningful patterns and trends.

## 2. Core Components <a name="core_components">

⚡ **Preprocessing and Transformation**: Comprehensive functions for extracting and preparing text data, ensuring it is ready for subsequent analysis.

⚡ **Feature Extraction**: Robust methods for generating `BoW`, `z-score` and `TF-IDF` matrices optionally for facilitating the conversion of text into meaningful numerical features.

⚡ **Dimensionality Reduction and Clustering**: Implementations of `PCA`, `SVD`, `MDS`, `t-SNE`, `k-Means`, `DBSCAN`, `HC` on choice to reduce data complexity and enable effective visualization of high-dimensional datasets.

⚡ **Visualization Capabilities**: Advanced plotting functions for visualizing the outcomes of dimensionality reduction, providing clear and actionable insights.
  * *It is important to know that the primary goal of the project was executing computations so are emphasized while the visualizations serve as a medium.*

⚡ **Customizable Configuration**: Modular design allowing for easy adjustment of settings and paths to accommodate diverse datasets and analysis requirements.

⚡ **Logging and Monitoring**: Integrated logging to track execution progress and capture potential issues, ensuring smooth and transparent operations.

## 3. Project Structure <a name="project_structure"></a>

```markdown
📁 project-root
├── 📁 config
│ ├── 📄 __init__.py
│ ├── 📄 common_config.py
│ ├── 📄 constants.py
│ └── 📄 nlp_config.py
│
├── 📁 src
│ ├── 📄 __init__.py
│ ├── 📄 preprocessing.py
│ ├── 📄 feature_extraction.py
│ ├── 📄 dimensionality_reduction.py
│ └── 📄 visualization.py
│
├── 📁 logs
│    ...
│
├── 📄 .gitignore
├── 📄 .gitattributes
└── 📄 main_script.py
```

---

* **config/**: Contains configuration files for the project.

  📜 ***__init__.py***: Imports configuration and utility functions.

  📜 ***common_config.py***: Contains common configuration functions and logging setup.

  📜 ***constants.py***: Defines constants used throughout the application.

  📜 ***nlp_config.py***: Handles NLP-specific configurations.


* **src/**: Contains source code for the project.
  💎 ***__init__.py***: Initializes the source package and imports functions from individual modules, setting up the namespace for the package.

  💎 ***preprocessing.py***: Handles text preprocessing tasks such as cleaning and preparing raw text data for further analysis.

  💎 ***feature_extraction.py***: Implements methods for extracting features from text data, including Bag of Words, z-score and TF-IDF.

  💎 ***dimensionality_reduction.py***: Contains functions for applying dimensionality reduction techniques to simplify and visualize high-dimensional data.

  💎 ***visualization.py***: Provides functions for generating visual representations of data, such as plots for dimensionality reduction and other analyses.

**logs**: Stores log files created during execution.

**.gitattributes**: Ensures consistent line endings across different operating systems in the repository.

**.gitignore**: Specifies files and directories to be ignored by Git.

**main.py**: Entry point for running scripts, if applicable.

## 4. Visualizations <a name="visualizations"></a>
Generates visualizations for dimensionality reduction results. Supports 1D, 2D, and 3D plots depending on the number of dimensions specified. The function saves the resulting plot to the given file path and logs the operation.

```python
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

```

### Reduction Graph `1D` <a name="graph_1"></a>

![Graph 1D](/screenshots/reduction_graph_1D.png?raw=true)

---

### Reduction Graph `2D` <a name="graph_2"></a>

![Graph 2D](/screenshots/reduction_graph_2D.png?raw=true)

---

### Reduction Graph `3D` <a name="graph_3"></a>

![Graph 3D](/screenshots/reduction_graph_3D.png?raw=true)


## 5. Usage <a name="usage"></a>
1. Ensure you have Python 3.x installed.

2. Install the required libraries and instances when needed.

3. Configure settings in config files as needed.

4. Run the scripts from the command line. Example:
```bash
python src/preprocessing.py
python src/feature_extraction.py
python src/dimensionality_reduction.py
python src/visualization.py

```

## 6. License <a name="license"></a>
This project is licensed under the GNU General Public License v3.0 (GPL-3.0) - see the [LICENSE](https://github.com/kivanc57/analytics_unsupervised/blob/main/LICENSE) file for details.

## 7. Contact <a name="contact"></a>
Feel free to reach out if you have any questions or suggestions!

* **Email**: kivancgordu@hotmail.com
* **Version**: 1.0.0
* **Date**: 05-08-2024



