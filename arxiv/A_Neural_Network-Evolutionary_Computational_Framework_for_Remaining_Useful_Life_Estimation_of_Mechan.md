# A Neural Network-Evolutionary Computational Framework for Remaining Useful Life Estimation of Mechanical Systems

**Authors:** David Laredo, Zhaoyin Chen, Oliver Schütze, Jian-Qiao Sun

**Published:** 2019-05-15

**Entry ID:** http://arxiv.org/abs/1905.05918v1

**Summary:** This paper presents a framework for estimating the remaining useful life (RUL) of mechanical systems. The framework consists of a multi-layer perceptron and an evolutionary algorithm for optimizing the data-related parameters. The framework makes use of a strided time window to estimate the RUL for mechanical components. Tuning the data-related parameters can become a very time consuming task. The framework presented here automatically reshapes the data such that the efficiency of the model is increased. Furthermore, the complexity of the model is kept low, e.g. neural networks with few hidden layers and few neurons at each layer. Having simple models has several advantages like short training times and the capacity of being in environments with limited computational resources such as embedded systems. The proposed method is evaluated on the publicly available C-MAPSS dataset, its accuracy is compared against other state-of-the art methods for the same dataset.

---

A Neural Network-Evolutionary Computational Framework
for Remaining Useful Life Estimation of Mechanical Systems
David Laredo1, Zhaoyin Chen1, Oliver Sch¨utze2 and Jian-Qiao Sun1
1Department of Mechanical Engineering
School of Engineering, University of California
Merced, CA 95343, USA
2Department of Computer Science, CINVESTAV
Mexico City, Mexico
Corresponding author. Email: davidlaredo1@gmail.com
Abstract
This paper presents a framework for estimating the remaining useful life (RUL) of
mechanical systems.
The framework consists of a multi-layer perceptron and an
evolutionary algorithm for optimizing the data-related parameters. The framework
makes use of a strided time window to estimate the RUL for mechanical components.
Tuning the data-related parameters can become a very time consuming task. The
framework presented here automatically reshapes the data such that the eﬃciency
of the model is increased. Furthermore, the complexity of the model is kept low, e.g.
neural networks with few hidden layers and few neurons at each layer. Having simple
models has several advantages like short training times and the capacity of being in
environments with limited computational resources such as embedded systems. The
proposed method is evaluated on the publicly available C-MAPSS dataset [1], its
accuracy is compared against other state-of-the art methods for the same dataset.
Keywords:
artiﬁcial neural networks, moving time window, RUL estimation,
prognostics, evolutionary algorithms
1. Introduction
Traditionally, maintenance of mechanical systems has been carried out based on
scheduling strategies. Such strategies are often costly and less capable of meeting the
increasing demand of eﬃciency and reliability [2, 3]. Condition based maintenance
(CBM) also known as intelligent prognostics and health management (PHM) allows
for maintenance based on the current health of the system, thus cutting down the
Preprint submitted to Elsevier
May 16, 2019
arXiv:1905.05918v1  [cs.LG]  15 May 2019
costs and increasing the reliability of the system [4]. Here, we refer to prognostics as
the estimation of remaining useful life of a system. The remaining useful life (RUL) of
the system can be estimated based on the historical data. This data-driven approach
can help optimize maintenance schedules to avoid engineering failures and to save
costs [5].
The existing PHM methods can be grouped into three diﬀerent categories: model-
based [6], data-driven [7, 8] and hybrid approaches [9, 10]. Model-based approaches
attempt to incorporate physical models of the system into the estimation of the
RUL. If the system degradation is modeled precisely, model-based approaches usually
exhibit better performance than data-driven approaches [11].
This comes at the
expense of having extensive a priori knowledge of the underlying system and having
a ﬁne-grained model of the system, which can involve expensive computations. On
the other hand, data-driven approaches use pattern recognition to detect changes in
system states. Data-driven approaches are appropriate when the understanding of
the ﬁrst principles of the system dynamics is not comprehensive or when the system
is suﬃciently complex such as jet engines, car engines and complex machineries, for
which it is prohibitively diﬃcult to develop an accurate model.
Common disadvantages for the data-driven approaches are that they usually ex-
hibit wider conﬁdence intervals than model-based approaches and that a fair amount
of data is required for training. Many data-driven algorithms have been proposed.
Good prognostics results have been achieved. Among the most popular algorithms
we can ﬁnd artiﬁcial neural networks (ANNs) [2], support vector machine (SVM)
[12], Markov hidden chains (MHC) [13] and so on. Over the past few years, data-
driven approaches have gained more attention in the PHM community. A number of
machine learning techniques, especially neural networks, have been applied success-
fully to estimate the RUL of diverse mechanical systems. ANNs have demonstrated
good performance in modeling highly nonlinear, complex, multi-dimensional systems
without any prior knowledge on the system behavior [14]. While the conﬁdence lim-
its for the RUL predictions cannot be analytically provided [15], the neural network
approaches are promising for prognostic problems.
Neural networks for estimating the RUL of jet engines have been previously ex-
plored in [16] where the authors propose a multi-layer perceptron (MLP) coupled
with a feature extraction (FE) method and a time window for the generation of the
features for the MLP. In the publication, the authors demonstrate that a moving
window combined with a suitable feature extractor can improve the RUL prediction
as compared with the studies with other similar methods in the literature. In [14],
the authors explore a deep learning ANN architecture, the so-called convolutional
neural networks (CNNs), where they demonstrate that by using a CNN without any
2
po