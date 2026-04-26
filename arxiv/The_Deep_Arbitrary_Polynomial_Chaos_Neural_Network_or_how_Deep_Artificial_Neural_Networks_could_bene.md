# The Deep Arbitrary Polynomial Chaos Neural Network or how Deep Artificial Neural Networks could benefit from Data-Driven Homogeneous Chaos Theory

**Authors:** Sergey Oladyshkin, Timothy Praditia, Ilja Kröker, Farid Mohammadi, Wolfgang Nowak, Sebastian Otte

**Published:** 2023-06-26

**Entry ID:** http://arxiv.org/abs/2306.14753v1

**Summary:** Artificial Intelligence and Machine learning have been widely used in various fields of mathematical computing, physical modeling, computational science, communication science, and stochastic analysis. Approaches based on Deep Artificial Neural Networks (DANN) are very popular in our days. Depending on the learning task, the exact form of DANNs is determined via their multi-layer architecture, activation functions and the so-called loss function. However, for a majority of deep learning approaches based on DANNs, the kernel structure of neural signal processing remains the same, where the node response is encoded as a linear superposition of neural activity, while the non-linearity is triggered by the activation functions. In the current paper, we suggest to analyze the neural signal processing in DANNs from the point of view of homogeneous chaos theory as known from polynomial chaos expansion (PCE). From the PCE perspective, the (linear) response on each node of a DANN could be seen as a $1^{st}$ degree multi-variate polynomial of single neurons from the previous layer, i.e. linear weighted sum of monomials. From this point of view, the conventional DANN structure relies implicitly (but erroneously) on a Gaussian distribution of neural signals. Additionally, this view revels that by design DANNs do not necessarily fulfill any orthogonality or orthonormality condition for a majority of data-driven applications. Therefore, the prevailing handling of neural signals in DANNs could lead to redundant representation as any neural signal could contain some partial information from other neural signals. To tackle that challenge, we suggest to employ the data-driven generalization of PCE theory known as arbitrary polynomial chaos (aPC) to construct a corresponding multi-variate orthonormal representations on each node of a DANN to obtain Deep arbitrary polynomial chaos neural networks.

---

THE DEEP ARBITRARY POLYNOMIAL CHAOS NEURAL
NETWORK OR HOW DEEP ARTIFICIAL NEURAL NETWORKS
COULD BENEFIT FROM DATA-DRIVEN HOMOGENEOUS CHAOS
THEORY
A PREPRINT
Sergey Oladyshkin
Department of Stochastic Simulation and Safety Research for Hydrosystems,
Institute for Modelling Hydraulic and Environmental Systems, Stuttgart Center for Simulation Science,
University of Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart, Germany
Sergey.Oladyshkin@iws.uni-stuttgart.de
Timothy Praditia
Department of Stochastic Simulation and Safety Research for Hydrosystems,
Institute for Modelling Hydraulic and Environmental Systems, Stuttgart Center for Simulation Science,
University of Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart, Germany
Ilja Kr¨oker
Department of Stochastic Simulation and Safety Research for Hydrosystems,
Institute for Modelling Hydraulic and Environmental Systems, Stuttgart Center for Simulation Science,
University of Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart, Germany
Farid Mohammadi
Department of Hydromechanics and Modelling of Hydrosystems,
Institute for Modelling Hydraulic and Environmental Systems,
University of Stuttgart, Pfaffenwaldring 61, 70569 Stuttgart, Germany
Wolfgang Nowak
Department of Stochastic Simulation and Safety Research for Hydrosystems,
Institute for Modelling Hydraulic and Environmental Systems, Stuttgart Center for Simulation Science,
University of Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart, Germany
Sebastian Otte
Neuro-Cognitive Modeling, Computer Science Department,
University of T¨ubingen, Sand 14, 72076 T¨ubingen, Germany
June 27, 2023
ABSTRACT
Artificial Intelligence and Machine learning have been widely used in various fields of mathemat-
ical computing, physical modeling, computational science, communication science, and stochastic
analysis. Approaches based on Deep Artificial Neural Networks (DANN) are very popular in our
days. Depending on the learning task, the exact form of DANNs is determined via their multi-layer
architecture, activation functions and the so-called loss function. However, for a majority of deep
learning approaches based on DANNs, the kernel structure of neural signal processing remains the
arXiv:2306.14753v1  [cs.NE]  26 Jun 2023
Deep Arbitrary Polynomial Chaos Artificial Neural Network
A PREPRINT
same, where the node response is encoded as a linear superposition of neural activity, while the
non-linearity is triggered by the activation functions. In the current paper, we suggest to analyze the
neural signal processing in DANNs from the point of view of homogeneous chaos theory as known
from polynomial chaos expansion (PCE). From the PCE perspective, the (linear) response on each
node of a DANN could be seen as a 1st degree multi-variate polynomial of single neurons from the
previous layer, i.e. linear weighted sum of monomials. From this point of view, the conventional
DANN structure relies implicitly (but erroneously) on a Gaussian distribution of neural signals. Ad-
ditionally, this view revels that by design DANNs do not necessarily fulfill any orthogonality or
orthonormality condition for a majority of data-driven applications. Therefore, the prevailing han-
dling of neural signals in DANNs could lead to redundant representation as any neural signal could
contain some partial information from other neural signals. To tackle that challenge, we suggest to
employ the data-driven generalization of PCE theory known as arbitrary polynomial chaos (aPC)
to construct a corresponding multi-variate orthonormal representations on each node of a DANN.
Doing so, we generalize the conventional structure of DANNs to Deep arbitrary polynomial chaos
neural networks (DaPC NN). They decompose the neural signals that travel through the multi-layer
structure by an adaptive construction of data-driven multi-variate orthonormal bases for each layer.
Moreover, the introduced DaPC NN provides an opportunity to go beyond the linear weighted su-
perposition of single neurons on each node. Inheriting fundamentals of PCE theory, the DaPC NN
offers an additional possibility to account for high-order neural effects reflecting simultaneous in-
teraction in multi-layer networks. Introducing the high-order weighted superposition on each node
of the network mitigates the necessity to introduce non-linearity via activation functions and, hence,
reduces the room for potential subjectivity in the modeling procedure. Although the current DaPC
NN framework has no theoretical restrictions on the use of activation functions. The current paper
also summarizes relevant properties of DaPC NNs inherited from aPC as analytical expressions for
statistical quantities and sensitivity indexes on each node. We also offer an analytical form of partial
derivatives that could be used in various training algorithms. Technically, DaPC NNs require similar
training procedures as conventional DANNs, and all trained weights determine automatically the
corresponding multi-variate data-driven orthonormal base