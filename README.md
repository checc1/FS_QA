# FS_QA
FS algorithm reformulated as a QUBO problem applied to feature maps extracted from a target layer of Computer vision model.

## Abstract
Deep learning models are used in critical applications where mistakes can have serious conse-
quences. Therefore, we must also understand how and why it makes that prediction. This under-
standing helps us check whether the model is learning the right patterns, detect biases in the data,
improve model design, and build systems that we can trust. This work proposes a new method
for interpreting Convolutional Neural Networks in image classification tasks. The approach works
by selecting the most important feature maps that contribute to each prediction. To solve this
combinatorial problem, we encode it into a quantum constrained optimization problem and pro-
pose to solve it using quantum annealing. We evaluate our method against the state-of-the-art
explainable AI techniques, specifically GradCAM and GradCAM++ and observe an improved class
disentanglement, i.e. the model’s decision boundaries become more distinct and its reasoning more
transparent. This demonstrates that our approach enhances the quality of explanations, making
it easier to understand which features the model relies on for specific predictions. In addition, we
study the computational behavior of the quantum annealing algorithm. Specifically, we analyze
the minimum energy gap of the system during computation and the probability that the algorithm
finds the correct solution. These analyses provide theoretical insight into why the method works
effectively in practice.

## Authors and contributors
Francesco Aldo Venturelli, Emanuele Costa, Sikha O K, Bruno Julia Diaz, Miguel A. Gonzalez Ballester and Alba Cervera-Lierta.

