

# **notes on papers**

# hastie2009elements

- Derivation for knn
- "The training error tends to decrease whenever we increase the model complexity, that is, whenever we fit the data harder. However with too much fitting, the model adapts itself too closely to the training data, and will not generalize well" (pg 38)

# belkin2019reconciling

- introduces double descent curve
- experiments with neural network and random fourier features shows double descent behavior
- talks about optimization being different in overparameterized setting, and acting as a regularizer

# zhang2017understanding

- neural networks easily fit random labels / easily have 100% training accuracy on data sets (randomization test)
- explicit regularizers have marginal benefit / SGD may act as an implicit regularizer (experimental evidence)
- kernel method acheives good results on MNIST with simple preprocessing

# neal2018modern

- decomposes variance in bias-variance decomposition further into variance caused by sampling and variance caused by optimization.
- finds that total variance decreases with very big models because variance caused by optimization decreases (experimental evidence)

# sejnowski2020unreasonable

- details history of neural networks in ML research
- details biological inspiration for neural networks, and possible future inspiration from biological brains

# belkin2021fit

- interpolating models often generalize well (weighted kNN example); maybe due to inductive bias: smoothest function is simplest or min norm solution
- PL\* condition suggest gradient descent converges in an overparameterized setting with MSE loss to a global maximum; SGD is also discussed
- Other stuff: MSE for classification? adversarial examples? early stopping? 

# rifkin2002everything

- from page 91: "To sum up, across a wide range of reasonable examples, Support Vector Machines and Regularized Least-Squares Classification perform very similarly. Both the classification functions and their performance are essentially identical."

# hui2020evaluation

- run a lot of classification experiments from NLP to CV, with both cross-entropy and square loss; square loss outperforms cross-entropy on all but computer vision

# janocha2017loss

- a few analyses of L1, L2 and Cauchy-Schwarz Divergence loss
- experiments with various loss functions show mixed results (for ANN w multiple layers, L2 is fast)

# kim2017convergence

- gives regret bounds for various optimization algorithms used for ANNs assuming convexity and L-Lipschitz continuous gradient

# masters2018revisiting

- compares different batch sizes and learning rates with SGD on CIFAR10 and CIFAR100, w and w out image augmentation, and batchnorm
- concludes that small batch sizes are better (and by looking at the graphs, 2^-5, which is about 3.1e-2, is a good learning rate)




# one day I'll get to these

- https://arxiv.org/pdf/2102.07238.pdf (Double-descent curves in neural networks: a new perspective using Gaussian processes)
- https://arxiv.org/pdf/1711.00165.pdf (DEEP NEURAL NETWORKS AS GAUSSIAN PROCESSES)
- https://proceedings.neurips.cc/paper/2019/file/ae614c557843b1df326cb29c57225459-Paper.pdf (On Lazy Training in Differentiable Programming)
- https://arxiv.org/pdf/2110.09485.pdf (Learning in High Dimension Always Amounts to Extrapolation)
- https://proceedings.neurips.cc/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf (Neural Tangent Kernel: Convergence and Generalization in Neural Networks)