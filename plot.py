from malwareDetector import malwareDetector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import logging
from prototypeLearning import get_original_prototype
from sklearn import manifold
import torch
import torch.nn as nn
from matplotlib.colors import ListedColormap
from textwrap import wrap
from sklearn.metrics import confusion_matrix

logger = logging.getLogger("dpc_cluster")


class plot():
    def __init__(self, detector: malwareDetector = malwareDetector(), bool = True):
        self.save_path = detector.picFolder
        self.detector = detector
        self.bool = bool

    def plot_cpuArch_tsne(self, tsne_data: np.array, cpu_list: np.array, y_labels: np.array, label_mapping: dict, title: str = "CpuArch t-SNE") -> None:
        '''
        Plot the original t-SNE.
        Args:
            tsne_data: The t-SNE data.
            title: The title of the plot.
        '''
        cpuType = list(set(cpu_list))
        cpu_to_marker = {cpu: marker for cpu, marker in zip(cpuType, ['o', 's', '^', 'P'])}

        tsne = TSNE(n_components=2, random_state=self.detector.seed)
        X_tsne = tsne.fit_transform(tsne_data)

        plt.figure(figsize=(12, 8))
        n_classes = len(set(y_labels))
        cmap = plt.get_cmap("tab20", n_classes)

        for i in label_mapping.keys():
            for cpu in cpuType:
                mask = (y_labels == i) & (cpu_list == cpu)
                plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                            color=cmap(i), marker=cpu_to_marker[cpu], 
                            facecolors='None',
                            label='_nolegend_')

        legend_elements = []
        for i in label_mapping.keys():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color=cmap(i), 
                                   label=label_mapping[i], markersize=8, linestyle='None'))

        legend_elements.append(plt.Line2D([0], [0], marker='', linestyle=''))
        
        for cpu in cpuType:
            legend_elements.append(plt.Line2D([0], [0], marker=cpu_to_marker[cpu], color='gray', 
                                   label=cpu, markersize=8, linestyle='None'))


        plt.legend(handles=legend_elements, title='Malware Classes & CPU Architectures', 
                   loc='center left', bbox_to_anchor=(1.05, 0.5))


        plt.title(f'Visualization of {title} in {self.detector.cpuArch}')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.grid(True)
        plt.tight_layout()
        if not os.path.exists(f"{self.save_path}/tsne"):
            self.detector.mkdir(f"{self.save_path}/tsne")
        plt.savefig(f"{self.save_path}/tsne/{title}_{self.detector.cpuArch}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_original_tsne(self, tsne_data: np.array, y_labels: np.array, label_mapping: dict, title: str = "Original t-SNE"):
        '''
        Plot the original t-SNE.
        Args:
            tsne_data: The t-SNE data.
            title: The title of the plot.
        '''
        tsne = TSNE(n_components=2, random_state=self.detector.seed)
        X_tsne = tsne.fit_transform(tsne_data)

        plt.figure(figsize=(12, 8))
        n_classes = len(set(y_labels))
        cmap = plt.get_cmap("nipy_spectral", n_classes)


        for i in label_mapping.keys():
            mask = y_labels == i
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=cmap(i), label=label_mapping[i])

        plt.legend(title='Malware Classes', loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.title(f'Visualization of {title}')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.grid(True)
        plt.tight_layout()
        if not os.path.exists(f"{self.save_path}/tsne"):
            self.detector.mkdir(f"{self.save_path}/tsne")
        plt.savefig(f"{self.save_path}/tsne/{title}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_tsne_cluster(self, tsne_data: np.array, cpu_list: np.array, family: str, clusterLabel: np.array) -> None:
        title = f"t-SNE {self.detector.cluster_method}"
        uniqueCluster = np.unique(clusterLabel)
        # get original prototype
        proto = get_original_prototype(tsne_data, clusterLabel)
        tsne_data = np.vstack((tsne_data, proto))

        cpuType = list(set(cpu_list))
        cpu_to_marker = {cpu: marker for cpu, marker in zip(cpuType, ['o', 's', '^', 'P'])}

        tsne = TSNE(n_components=2, random_state=self.detector.seed)
        X_tsne = tsne.fit_transform(tsne_data)

        X_data = X_tsne[:-len(proto)]
        X_proto = X_tsne[-len(proto):]

        plt.figure(1, figsize=(10, 6))
        n_classes = len(set(clusterLabel))
        cmap = plt.get_cmap("tab20", n_classes)

        for i, label in enumerate(uniqueCluster):
            plt.scatter(X_proto[i, 0], X_proto[i, 1], color=cmap(i), marker='X', s=200, edgecolors='black', linewidth=0.5, label=f'Prototype {uniqueCluster[i]}')
            for cpu in cpuType:
                mask = (clusterLabel == uniqueCluster[i]) & (cpu_list == cpu)
                plt.scatter(X_data[mask, 0], X_data[mask, 1], 
                            color=cmap(i), marker=cpu_to_marker[cpu], 
                            facecolors='None',
                            label='_nolegend_')
            

        legend_elements = []

        for i in range(len(uniqueCluster)):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color=cmap(i), 
                                   label="Cluster " + str(uniqueCluster[i]), markersize=8, linestyle='None'))

        legend_elements.append(plt.Line2D([0], [0], marker='', linestyle=''))
        
        for cpu in cpuType:
            legend_elements.append(plt.Line2D([0], [0], marker=cpu_to_marker[cpu], color='gray', 
                                   label=cpu, markersize=8, linestyle='None'))


        plt.legend(handles=legend_elements, title='Malware Classes & CPU Architectures', 
                   loc='center left', bbox_to_anchor=(1.05, 0.5))


        plt.title(f'Visualization of {title} in {family}')
        plt.xlabel('t-SNE Feature 1')
        plt.ylabel('t-SNE Feature 2')
        plt.grid(True)
        plt.tight_layout()
        if not os.path.exists(f"{self.save_path}/tsne"):
            os.mkdir(f"{self.save_path}/tsne")
        plt.savefig(f"{self.save_path}/tsne/{title}_{self.detector.cpuArch}_{family}.png", dpi=300, bbox_inches='tight')
        plt.show()       


    def plot_rho_delta(self, rho, delta, family = None):
        '''
        Plot scatter diagram for rho-delta points

        Args:
            rho   : rho list
            delta : delta list
        '''
        logger.info("PLOT: rho-delta plot")
        self.plot_scatter_diagram(0, rho[1:], delta[1:], x_label='rho', y_label='delta', title=f'Decision Graph {family}', )


    def plot_cluster(self, cluster, family = None):
        '''
        Plot scatter diagram for final points that using multi-dimensional scaling for data

        Args:
            cluster : DensityPeakCluster object
        '''
        logger.info("PLOT: cluster result, start multi-dimensional scaling")
        dp = np.zeros((cluster.max_id, cluster.max_id), dtype = np.float32)
        cls = []
        for i in range(1, cluster.max_id):
            for j in range(i + 1, cluster.max_id + 1):
                dp[i - 1, j - 1] = cluster.distances[(i, j)]
                dp[j - 1, i - 1] = cluster.distances[(i, j)]
            cls.append(cluster.cluster[i])
        cls.append(cluster.cluster[cluster.max_id])
        cls = np.array(cls, dtype = np.float32)
        fo = open(f'./{self.detector.clusterResultFolder}/{self.detector.cpuArch}/{family}_cluster.txt', 'w')
        fo.write('\n'.join(map(str, cls)))
        fo.close()
        mds = manifold.MDS(max_iter=200, eps=1e-4, n_init=1,dissimilarity='precomputed', random_state=self.detector.seed)
        dp_mds = mds.fit_transform(dp.astype(np.float64))
        logger.info("PLOT: end mds, start plot")
        self.plot_scatter_diagram(1, dp_mds[:, 0], dp_mds[:, 1], title=f'2D Nonclassical Multidimensional Scaling {family}', style_list = cls)


    def plot_rhodelta_rho(self, rho, delta, family = None):
        '''
        Plot scatter diagram for rho*delta_rho points

        Args:
            rho   : rho list
            delta : delta list
        '''
        logger.info("PLOT: rho*delta_rho plot")
        y=rho*delta
        r_index=np.argsort(-y)
        x=np.zeros(y.shape[0])
        idx=0
        for r in r_index:
            x[r]=idx
            idx+=1
        plt.figure(2)
        plt.clf()
        plt.scatter(x,y)
        plt.xlabel('sorted rho')
        plt.ylabel('rho*delta')
        plt.title(f"Decision Graph RhoDelta-Rho {family}")
        plt.savefig(f'{self.save_path}/{self.detector.cpuArch}/Decision Graph RhoDelta-Rho {family}.jpg', dpi=300)
        plt.show()

    def plot_scatter_diagram(self, which_fig, x, y, x_label = 'x', y_label = 'y', title = 'title', style_list = None, family = None):
        '''
        Plot scatter diagram

        Args:
            which_fig  : which sub plot
            x          : x array
            y          : y array
            x_label    : label of x pixel
            y_label    : label of y pixel
            title      : title of the plot
        '''
        styles = ['k', 'g', 'r', 'c', 'm', 'y', 'b', '#9400D3','#C0FF3E']
        assert len(x) == len(y)
        if style_list is not None:
            assert len(x) == len(style_list) and len(styles) >= len(set(style_list))
        plt.figure(which_fig)
        plt.clf()
        if style_list is None:
            # plt.plot(x, y, color=styles[0], linestyle='.', marker='.')
            plt.scatter(x, y, color=styles[0], marker='.')
            for i, (xi, yi) in enumerate(zip(x, y)):
                plt.annotate(f'({xi:.2f}, {yi:.2f})', (xi, yi), textcoords='offset points', xytext=(0, 10), ha='center')
        else:
            clses = set(style_list)
            xs, ys = {}, {}
            for i in range(len(x)):
                try:
                    xs[style_list[i]].append(x[i])
                    ys[style_list[i]].append(y[i])
                except KeyError:
                    xs[style_list[i]] = [x[i]]
                    ys[style_list[i]] = [y[i]]
            added = 1
            for idx, cls in enumerate(clses):
                if cls == -1:
                    style = styles[0]
                    added = 0
                else:
                    style = styles[idx + added]
                # plt.plot(xs[cls], ys[cls], color=style, linestyle='.', marker='.')
                plt.scatter(xs[cls], ys[cls], color=style, marker='.')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(f'{self.save_path}/{self.detector.cpuArch}/{title}.jpg', dpi=300)
        plt.show()

    def plot_visualize_embeddings(self, prototypes, query_embeddings, query_labels, label_mapping, pred_y, original_label = None):
        prototypes_np = prototypes.detach().numpy()
        n_way = prototypes_np.shape[0]
        # print(prototypes_np.shape)
        # try:
        #     query_labels = query_labels.squeeze(2)
        #     query_labels = query_labels.reshape(-1)
        # except:
        #     pass

        _, n_query = np.unique(query_labels, return_counts=True)
        if self.detector.cluster:
            ans = torch.arange(0, query_labels.unique().shape[0])
            ans = np.repeat(ans, n_query)

        query_embeddings_np = query_embeddings.detach().numpy()


        # Combine prototypes and query embeddings for t-SNE
        combined_embeddings = np.vstack((prototypes_np, query_embeddings_np))
        tsne = TSNE(n_components=2, random_state=self.detector.seed)
        embeddings_2d = tsne.fit_transform(combined_embeddings)
        # Separate the transformed embeddings
        prototypes_2d = embeddings_2d[:n_way]
        query_embeddings_2d = embeddings_2d[n_way:]

        cmap = plt.get_cmap("tab20")
        
        # Plotting
        plt.figure(figsize=(14, 8))
        label_index = 0
        for i in label_mapping.keys():
            if self.detector.cluster:
                family = original_label[i]
                query_1 = np.sum(n_query[:label_index])
                query_2 = np.sum(n_query[:label_index+1])
                for j, index in enumerate(label_mapping[i]):
                    plt.scatter(prototypes_2d[index, 0], prototypes_2d[index, 1], marker='X', s=200, color=cmap(label_index), edgecolors='black', linewidth=0.5, label="\n".join(wrap(f'Prototype {family}:{index}', 20)))
                plt.scatter(query_embeddings_2d[query_1:query_2, 0], query_embeddings_2d[query_1:query_2, 1],marker='o', facecolors='none', color=cmap(label_index),linewidths=0.5, label="\n".join(wrap(f'Query {family}', 20)))
                label_index += 1
            else:
                plt.scatter(prototypes_2d[i, 0], prototypes_2d[i, 1], marker='X', s=200, color=cmap(i), edgecolors='black', linewidth=0.5, label="\n".join(wrap(f'Prototype {label_mapping[i]}', 20)))
                query_1 = np.sum(n_query[:i])
                query_2 = np.sum(n_query[:i+1])
                plt.scatter(query_embeddings_2d[query_1:query_2, 0], query_embeddings_2d[query_1:query_2, 1], marker='o', facecolors='None', color=cmap(i),linewidths=0.5,  label="\n".join(wrap(f'Query {label_mapping[i]}', 20)))
        
        if self.detector.cluster:
            for i in range(ans.shape[0]):
                if ans[i] != pred_y[i]:
                    print(f"Query {ans[i]} is misclassified as {pred_y[i]} at index {i}")
                    plt.scatter(query_embeddings_2d[i, 0], query_embeddings_2d[i, 1], marker='x', s=100, color=cmap(pred_y[i]), linewidth=0.5)
        else:
            for i in range(query_labels.shape[0]):
                if query_labels[i] != pred_y[i]:
                    print(f"Query {query_labels[i]} is misclassified as {pred_y[i]} at index {i}")
                    plt.scatter(query_embeddings_2d[i, 0], query_embeddings_2d[i, 1], marker='x', s=100, color=cmap(pred_y[i]), linewidth=0.5)

        plt.grid(True)
        plt.legend(ncol=2, bbox_to_anchor=(1, 1), loc='upper left')
        plt.title('t-SNE Visualization of Prototypes and Queries')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.savefig(f"{self.save_path}/tsne/{self.detector.model_name}.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        '''
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        '''
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        # print(y_true.shape)
        # print(y_pred.shape)
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = [v for k, v in classes.items()]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True label',
            xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)
        fig.tight_layout()
        if not os.path.exists(f"{self.save_path}/confusionMatrix"):
            self.detector.mkdir(f"{self.save_path}/confusionMatrix")
        plt.savefig(f"{self.save_path}/confusionMatrix/{self.detector.model_name}.png", dpi=300, bbox_inches='tight')
        plt.show()