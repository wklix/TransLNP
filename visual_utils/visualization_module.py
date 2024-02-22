from io import StringIO
import logging
from utils import logger
import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr
from rdkit import Chem
import rdkit.Chem.Draw as Draw
import matplotlib
from matplotlib import pyplot as plt
import umap
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
class YourVisualizationClass:
    def _eval_stratified_classes(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        q=6,
        use_set="all",
    ):
        """
        Evaluate the stratified classes of the predictions. If labels and predictions
        are intergers, will directly view them as categories. If labels and predictions
        are floats, will first convert them to categories by stratifying them.

        The stratification is done by sorting the predictions and labels in descending
        order and then split them into 5 groups with equal number of samples based
        on the quantiles.

        Args:
            labels (np.ndarray): Labels of the samples
            predictions (np.ndarray): Predictions of the samples
            q (int): Number of quantiles to split the samples into. Defaults to 4.
            use_set (str): The set to evaluate on, choices from train, test, all.
                Defaults to "test".
        """

        # fmt: off
        #logger.info("labels: \n{}".format(labels))
        #logger.info("predictions: \n{}".format(predictions))
        # if labels and predictions are floats, convert them to categories
        if not isinstance(labels[0], int) and not isinstance(predictions[0], int):
            labels = pd.qcut(labels, q, labels=False, duplicates="drop")
            predictions = pd.qcut(predictions, q, labels=False, duplicates="drop")
        #logger.info("labels: \n{}".format(labels))
        #logger.info("predictions: \n{}".format(predictions))
        # calculate the number of samples in each category
        num_samples = len(labels)
        num_samples_in_each_category = [
            len(labels[labels == i]) for i in np.unique(labels)
        ]
        assert len(labels) == len(predictions)
        #print(
        #    "Number of samples in each category:",
        #    num_samples_in_each_category,
        #    "Total number of samples:",
        #    num_samples,
        #)
        acc = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="macro")
        recall = recall_score(labels, predictions, average="macro")
        f1_macro = f1_score(labels, predictions, average="macro")
        f1_micro = f1_score(labels, predictions, average="micro")
        #print(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}")
        #print(f"Recall: {recall:.4f}, F1_macro: {f1_macro:.4f}, F1_micro: {f1_micro:.4f}")
        self.stratified_class_results = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "macro f1": f1_macro,
            "micro f1": f1_micro,
        }

    #    import csv
        # specify the file name
    #    filename = os.path.join(f"matrics111_{use_set}.csv" )

        # writing to csv file
    #    with open(filename, 'w') as csvfile:
            # creating a csv writer object
    #        csvwriter = csv.writer(csvfile)

            # writing the headers
    #        csvwriter.writerow(self.stratified_class_results.keys())

            # writing the data rows
    #        csvwriter.writerow(self.stratified_class_results.values())


        # compute and plot confusion matrix using seaborn api
        labels_to_show = np.sort(np.unique(labels))
        #logger.info("labels_to_show: \n{}".format(labels_to_show)) 
        num_cates = len(labels_to_show)
        cm = confusion_matrix(labels, predictions, labels=labels_to_show)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        #logger.info("cm: \n{}".format(cm)) 
        import seaborn as sns
        sns.set_style("white")
        sns.set_context("paper", font_scale=1.5)
        fig, ax = plt.subplots(figsize=(10, 8))
        cmdata = cm.tolist()
        heatmap =sns.heatmap(
            cmdata,
            annot=True,
            cmap="Blues",
            fmt=".2f",
            ax=ax
        )
        #print(cm)
        #logger.info("type: \n{}".format(type(cm))) 
        # ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted top k percentiles")
        ax.set_ylabel("Actual top k percentiles")
        # set the stick postions at num_cates + 1 positions
        ax.set_xticks(np.linspace(0, num_cates, num_cates + 1))
        ax.set_yticks(np.linspace(0, num_cates, num_cates + 1))
        # set the tick labels
        ax.set_xticklabels(
            [
                f"{i}%" if i != 0 else "Top"
                for i in np.linspace(100, 0, num_cates + 1).astype(int)
            ]
        )
        ax.set_yticklabels(
            [
                f"{i}%" if i != 0 else ""
                for i in np.linspace(100, 0, num_cates + 1).astype(int)
            ]
        )
        #logger.info("set_yticklabels: \n{}".format(   [
        #        f"{i}%" if i != 0 else ""
        #        for i in np.linspace(100, 0, num_cates + 1).astype(int)
        #    ])) 
        #logger.info("set_xticklabels: \n{}".format([
        #        f"{i}%" if i != 0 else "Top"
        #        for i in np.linspace(100, 0, num_cates + 1).astype(int)
        #    ])) 
        self.stratified_class_results["confusion matrix"] = cm
        self.stratified_class_results["confusion matrix fig"] = fig

        # save the confusion matrix
        fig.savefig(

                f"./visual_utils/confusion_matrix_{use_set}.png"
        )


    def visualize(
            self,
            embeddings: np.ndarray,
            labels: np.ndarray = None,
            predictions: np.ndarray = None,
            color_key: str = "labels",
        ) -> matplotlib.figure.Figure:
            """
            Visualize the embeddings with UMAP

            Args:
                embeddings (np.ndarray): Raw embeddings to visualize
                labels (np.ndarray, optional): Labels of the embeddings
                predictions (np.ndarray, optional): Predictions values. Defaults to None.
                color_key (str): Use which field to color the points. Defaults to "labels".

            Returns:
                matplotlib.figure.Figure
            """
            #print("Embeddings shape:", embeddings.shape)

            if color_key == "labels":
                color = labels
                legend_name = "Label Efficiency"
            elif color_key == "predictions":
                color = predictions
                legend_name = "Predicted efficiency"

            if labels is not None and predictions is not None:
                self._eval_stratified_classes(labels, predictions)

            if predictions is not None and labels is not None:
                #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
                fig, ax1 = plt.subplots(figsize=(12, 10))
                

            # umap visualization of embeddings with hue as labels
            # reducer = umap.UMAP(n_neighbors=30, min_dist=1.0, spread=1.0)
            # reducer = umap.UMAP(n_neighbors=60, min_dist=1.0, spread=1.0)
            # reducer = umap.UMAP(n_neighbors=60, min_dist=1.0, spread=1.0, metric="cosine")
            reducer = umap.UMAP(
                n_neighbors=60,
                min_dist=1.0,
                spread=1.0,
                metric="cosine",
                random_state=12,
            )
            embedding = reducer.fit_transform(embeddings)
            #embedding = reducer.fit_transform(embeddings.cpu().numpy())
            self.umap_emb = embedding
            # customize cmap
            # cmap_ = matplotlib.cm.get_cmap("Accent_r")
            # make a cmap using three key colors
            cmap_ = matplotlib.colors.LinearSegmentedColormap.from_list(
                "mycmap",
                # ["#44548c", "#5a448e", "#377280"],
                # ["#5f6da0", "#735fa2", "#518692"],
                # ["#5a448e", "#735fa2", "#cec6e1", "#bed5db", "#377280"],
                 ["#e6a23c", "#fbbf60", "#fdcc89", "#fddbb1", "#fee9d9"],
                # ["#5a448e", "#907fb7", "#709d7a", "#377280"],
                #["#cec6e1", "#bda3cd", "#907fb7", "#735fa2", "#5a448e"],
            )
            im = ax1.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=color,
                s=60 * 1000 / len(color),
                #cmap=cmap_,
                cmap="Spectral",
                alpha=0.9,
            )
            # remove ticks and spines
            ax1.set(xticks=[], yticks=[])
            #ax1.set_title("UMAP projection of the dataset", fontsize=24)
            ax1.spines["top"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["bottom"].set_visible(False)
            ax1.spines["left"].set_visible(False)
            cbar = fig.colorbar(
                im,
                ax=ax1,
                # ticks=[i for i in np.linspace(np.min(color), np.max(color), 5)],
                format="%.1f",
                orientation="vertical",
                shrink=0.5,
            )
            cbar.ax.tick_params(labelsize=20)
            cbar.ax.set_ylabel(legend_name, rotation=270, fontsize=20, labelpad=20)



            return fig
