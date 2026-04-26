# MECCH: Metapath Context Convolution-based Heterogeneous Graph Neural Networks

**Authors:** Xinyu Fu, Irwin King

**Published:** 2023-11-23

**Entry ID:** http://arxiv.org/abs/2211.12792v2

**Summary:** Heterogeneous graph neural networks (HGNNs) were proposed for representation learning on structural data with multiple types of nodes and edges. To deal with the performance degradation issue when HGNNs become deep, researchers combine metapaths into HGNNs to associate nodes closely related in semantics but far apart in the graph. However, existing metapath-based models suffer from either information loss or high computation costs. To address these problems, we present a novel Metapath Context Convolution-based Heterogeneous Graph Neural Network (MECCH). MECCH leverages metapath contexts, a new kind of graph structure that facilitates lossless node information aggregation while avoiding any redundancy. Specifically, MECCH applies three novel components after feature preprocessing to extract comprehensive information from the input graph efficiently: (1) metapath context construction, (2) metapath context encoder, and (3) convolutional metapath fusion. Experiments on five real-world heterogeneous graph datasets for node classification and link prediction show that MECCH achieves superior prediction accuracy compared with state-of-the-art baselines with improved computational efficiency.

---

MECCH: Metapath Context Convolution-based Heterogeneous Graph Neural Networks
Xinyu Fua, Irwin Kinga
aDepartment of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China
Abstract
Heterogeneous graph neural networks (HGNNs) were proposed for representation learning on structural data with multiple types
of nodes and edges. To deal with the performance degradation issue when HGNNs become deep, researchers combine metapaths
into HGNNs to associate nodes closely related in semantics but far apart in the graph. However, existing metapath-based models
suffer from either information loss or high computation costs. To address these problems, we present a novel Metapath Context
Convolution-based Heterogeneous Graph Neural Network (MECCH). MECCH leverages metapath contexts, a new kind of graph
structure that facilitates lossless node information aggregation while avoiding any redundancy. Specifically, MECCH applies three
novel components after feature preprocessing to extract comprehensive information from the input graph efficiently: (1) metap-
ath context construction, (2) metapath context encoder, and (3) convolutional metapath fusion. Experiments on five real-world
heterogeneous graph datasets for node classification and link prediction show that MECCH achieves superior prediction accuracy
compared with state-of-the-art baselines with improved computational efficiency. The code is available at https://github.com/
cynricfu/MECCH. The formal publication is available at https://doi.org/10.1016/j.neunet.2023.11.030.
Keywords: Graph neural networks, Heterogeneous information networks, Graph representation learning
1. Introduction
Many real-world networks are heterogeneous graphs, which
contain multiple types of nodes and edges. As illustrated in
Figure 1, a movie information network may consist of actors,
movies, directors, and different types of relationships between
them. The complex and irregular interactions among different
types of nodes and edges make it challenging to extract knowl-
edge from heterogeneous graphs efficiently. Therefore, hetero-
geneous graph representation learning, which aims to represent
nodes using low-dimensional vectors, is a desirable way to au-
tomatically process and make inferences on such data.
Over the past decade, heterogeneous graph representation
learning has drawn significant attention. Early attempts usually
combine skip-gram model (Mikolov et al., 2013a) and metapath-
guided random walks (Dong et al., 2017; Fu et al., 2017; Shi
et al., 2019). With the rapid development of deep learning,
graph neural networks (GNNs) (Kipf and Welling, 2017; Hamil-
ton et al., 2017; Velickovic et al., 2018) are proposed to incor-
porate node features and benefit from neural network architec-
tures. Initially, GNNs focused on homogeneous graphs. But it
is straightforward for researchers to generalize GNNs to the het-
erogeneous scenario, where multiple types of nodes and edges
introduce another layer of complexity into the GNN design.
Recent efforts in heterogeneous GNNs (HGNNs) can be di-
vided into two categories: relation-based HGNNs and metapath-
based HGNNs. Relation-based HGNNs consider message pass-
ing parameterized by edge types and aggregate information from
direct neighbors (Schlichtkrull et al., 2018; Zhang et al., 2019a;
Hu et al., 2020; Lv et al., 2021; Yang et al., 2023d). Mod-
els of this kind are generally simple and fast. Still, they usu-
Actor 1
Actor 2
Actor 3
Movie 1
Movie 2
Movie 3
Director 1
Director 2
Metapath
Instance
Metapath
Context
Metapath
Metapath-guided Neighbor
Figure 1: An illustration of the heterogeneous graph and related concepts.
ally require stacking many GNN layers to leverage informa-
tion multiple hops away, potentially deteriorating model per-
formance (Li et al., 2018; Zhou et al., 2021). Another line of
research takes advantage of metapaths, which are ordered se-
quences of node and edge types describing composite relation-
ships between nodes (as shown in Figure 1). Metapath-based
HGNNs consider message passing through metapaths and ag-
gregate information from metapath-guided neighbors. Through
this way, the models can comfortably obtain information mul-
tiple hops away with very few layers and capture high-level se-
mantic information embedded in the graph.
However, existing metapath-based HGNNs can hardly achieve
a good balance between model performance and computational
efficiency.
HAN (Wang et al., 2019a) aggregates metapath-
guided neighbors, but with information loss caused by discarded
intermediate nodes along metapaths. MAGNN (Fu et al., 2020)
addresses this issue by encoding metapath instances. But the
Preprint submitted to Neural Networks
November 27, 2023
arXiv:2211.12792v2  [cs.LG]  23 Nov 2023
computational complexity is significantly increased due to re-
dundant computations introduced. Unlike HAN and MAGNN,
which choose metapaths based on human expertise, GTN (Yun
et al., 2019) tries to automatically sel