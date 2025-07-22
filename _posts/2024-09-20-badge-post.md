---
title: "BADGE: A promising Active learning method"
date: 2024-09-20
---
The promise of Active Learning has always appeared to be a mirage when practising deep learning in industrial settings.

While it is touted as one of the first steps to take in the data collection phase of machine learning, many active learning methods only perform as-well-as if not marginally better than random sampling of data for annotation in many real-world datasets.

Unfortunately, most active-learning methods often show nearly similar performance under the ideal testing standards that are currently used in academia. 

> Benchmark datasets like CIFAR-10, SVHN and ImageNet-1k, while helpful for fair comparison amongst different algorithms over the years, are often well-balanced and have been a constant target for optimization.

 Constant use of the same benchmarks, in the long-term, often result in development of algorithms that essentially over-fit to the patterns inherent in the dataset. Thus, their performance falls well short of expectations in practise.

**BADGE: Batch Active Learning by Diverse Gradient Embeddings** is one of few active learning algorithms whose performance is robust in real-world settings. While they do test only on the standard benchmark datasets (CIFAR-10, SVHN, OpenML), the tests are noticeably exhaustive as they conduct 33 experiments for each algorithm that they compared with.

> The key concept in BADGE is that new samples are selected with a probability that’s proportional to their squared distance from the nearest pre-selected sample in the model’s gradient embedding of samples. 
{: .pull-content}

> BADGE eliminates a failure mode of Uncertainty-based active learning methods(Conf, Marg, Entropy).

These methods may select several identical, uncertain samples, for training, when just one sample may be sufficient to alleviate the model’s uncertainty. This results in a narrow range of exposure to the data distribution. It reduces the potential for a significant update of parameters that’s possible with diverse and difficult samples. Such methods sometimes perform poorer than training datasets constructed with random sampling.

The gradient embedding, for a sample, is the gradient of the cross-entropy loss with respect to the weights of the penultimate layer in the neural network. The loss is computed with the assumption that the class with the highest confidence is the true label.

# BADGE Algorithm

> 1. Initially select M random samples to create the labelled set.
> 2. Train the model on the current labelled set by minimizing the cross-entropy loss $L_CE$
> 3. For each sample x,  compute their gradient embedding with the hypothetical label (highest confident class)
> 4. Select a subset of samples using the K-means++ seeding algorithm on the computed gradient embeddings
> 5. Repeat steps 2-4 until the desired model performance is reached after T iterations

$$
L_{CE}(f(x;\theta),y)= -\sum_{i=1}^CI(y=i)\text{ln}(p_y)
$$

# Why BADGE?

The reason for BADGE’s robustness is likely explained by the following observations:

1. The gradient embedding, induced by a sample, is the lower bound on the possible gradient update for the model’s parameters if the sample is labelled for training. Thus the influence of the selected sample is likely to be higher on the model’s parameters.

2. It optimizes for diversity and uncertainty of samples using the gradient embeddings.

The first observation is illustrated in Fig. 1 where, for an unlabeled sample x, the gradient vector $g_x$ is calculated with the assumption that y is the class with highest softmax score (‘Car’). Once the sample is labelled, the l2 norm of the gradient vector $g_x$  is higher when the true label y is known to be any of the other classes (‘Plane’, ‘Train’, ‘Bike’). This could be intuitively interpreted as a greater magnitude or number of parameters needing to be changed since the model estimated low confidence scores for the true class as shown in Fig. 1.

Assuming the the true class is given by the class with the highest confidence:

$$
\hat{y}=argmax_{i\in[C]} f(x;\theta)
$$

where the softmax output of the network for an input is given by

$$
f(x;\theta)=\sigma(W.z(x;V))
$$

Sigma is the softmax operator, W is the weight matrix of the last layer and V consists of the weights of all the previous layers. The z operator is the non-linear function that maps an input x to the output of the network’s penultimate layer.

The gradients for the i-th block i.e parameters that compute the logits for the i-th label is given by

$$
(g_x)_i=\frac{\partial}{\partial{W_i}} L_{\text{CE}}(f(x;\theta),\hat{y})=(f(x;\theta)_i-I(\hat{y}=i))z(x;V)
$$ 

Where $L_CE$ is the cross-entropy loss with respect to the hypothetical true label. From the above derivation, we realise that each block of g$_x$ is a scaling of z(x; V ), which is the output of the penultimate layer of the network. In this respect, $g_x$ captures x’s representation information.
The above formula shows that if the i-th block is the hypothetical true label then its gradients would be scaled by a factor of its (softmax score - 1) where I is the indicator function, otherwise it would be scaled by just the softmax score.

<img src="{{ '/assets/images/gradient_badge.jpg' | relative_url }}" alt="Explanation of the BADGE method">
Fig.1 - The gradient norm on the unlabeled sample is less than or equal to the gradient norm on the labelled sample. Thus the change in parameters caused by the selected sample is often higher than the conservative estimate.



> The first observation guarantees some update of the weights since it’s a conservative estimate of the sample’s influence on the model.

> The second observation could be explained by the combined use of the gradient vector and the K-means++ seeding algorithm to optimize for uncertainty and diversity respectively. 

The model is considered uncertain about a sample x if knowing the true label induces a large gradient of the loss with respect to the model parameters thereby causing a large update to the model.The K-means++ seeding algorithm samples batches with high diversity. It mitigates the issue of uncertainty based sampling in which k identical samples are chosen due to their high uncertainty scores. The gradient embeddings with the highest norms are usually ones with high entropy (high uncertainty) and are more likely to be selected since they’re farther to the other embeddings. Conversely, the samples with highly confident scores have gradient embeddings with a small norm. They’re unlikely to be repeatedly selected by K-means++ at a specific iteration thus preventing well-known samples from being selected in the subsequent labeling iterations.Thus, K-Means++ initialisation algorithm ensures that a diverse set of high gradient-magnitude samples are selected for labelling with the algorithm detailed below:

# K-Means++ Seeding algorithm 

> 1. Select a random point as the initial centroid in the set K
> 2. Calculate the l2 distance D(x) of all the remaining points from their closest neighbor in K
> 3. Select a point with the probability $ D(x)^2/ Sum(D(xi)^2) $ and add it as a centroid to the set K
> 4. Repeat steps 2-4, K-1 times

The K-means++ algorithm is remarkable in that it sets an initial state where the centroids are far apart increasing the chance to approximate well-separated clusters and it provides an O(log K) approximation guarantee in expectation of the total error, consistently.

# Related Methods:

1. CORESET: A diversity-based approach using coreset selection. The embedding of each example is computed by the network’s penultimate layer and the samples at each round are selected using a greedy furthest-first traversal conditioned on all labeled examples

2. CONF (Confidence Sampling): An uncertainty-based active learning algorithm that selects B examples with smallest predicted class probability

3. MARG (Margin Sampling): An uncertainty-based active learning algorithm that selects the bottom B examples sorted according to the example’s mult-iclass margin between the first and second highest classes for a sample x

4. ENTROPY: An uncertainty-based active learning algorithm that selects the top B examples according to the entropy of the example’s predictive class probability distribution

5. ALBL (Active Learning by Learning): A bandit-style meta-active learning algorithm that selects between CORESET and CONF at every round

6. RAND : The naive baseline of randomly selecting k examples to query at each round

# Results

The experiments are conducted in a robust setting with 3 batch sizes(B) and 11 dataset(D)-architecture(A) pairs making the total number of (D,B,A) combinations to n=33. The labelling budget(L) is selected such that the learning is still progressive since all the algorithms eventually reach a similar performance for larger labelling budgets.  The experiments were conducted with a starting random dataset of 100 labelled samples. Each experiment is repeated 5 times and the average test error is used.

<img src="{{'/assets/imagesbadge_comp_matrix.jpg' | relative_url}}" alt="Comparison of BADGE with other active learning method">

Fig. 2 - A pairwise penalty matrix over all experiments. Element P(i,j) corresponds roughly to the number of times algorithm ‘i’ outperforms algorithm ‘j’. Column-wise averages at the bottom show overall performance (lower is better).

In the pairwise penalty matrix in Fig.2, BADGE generally outperforms all baselines. The matrix is constructed by accumulating a penalty of $1 / n_D,B,A$ for an element $P_{i,j}$ when algorithm i beats algorithm j in an experiment. The two-sided t-test is applied to each experiment on a set of 5 test errors (5 repeated experiments) for each algorithm. The two-sided t-test essentially determines if the mean error of either algorithm is the same, greater than or less than the other (p-value being 0.05).

<img src="{{'/assets/images/cdf_norm_errors.jpg' | relative_url}}" alt="Plot of the CDF of normalised errors">

Fig. 3 - The cumulative distribution function of normalised errors. The higher the CDF value, the better the performance of the algorithm.

In Fig. 3 , for a value of x the y value is the total no. of settings where the algorithm has a normalised error of at most x. BADGE & MARG are the most robust in most of the experiments as the normalised error is at most 1.2 for all its experiment settings(y=1.0) whereas only 82-90% of experiments for most other algorithms have errors limited to 1.2.

 The normalised error is defined as 

$$ ne_i=\bar{e}_i/\bar{e}_r $$

where r is the index of the Random sampling based experiments that are used to normalise all the other sampling methods.

$$ \bar{e}_i=\frac{1}{5}\sum_{l=1}^5e_{i}^{l} $$

is the average error for each algorithm i.

# Summary

1. BADGE is a practical, easy to implement active learning algorithm that has robust empirical evidence to back its performance.

2. The use of gradient embeddings with hypothetical labels guarantees selecting samples with higher influences on the model.

3 K-means++ seeding algorithm optimises for intra-batch diversity and uncertainty in selecting the samples  

# Thoughts

1. Testing with highly skewed datasets with fine-grained classes would be a significant robustness test for BADGE, BALD.

2. Experiments on pre-trained, self-supervised and semi-supervised models should be conducting to determine if active learning is useful on top of these currently popular methods.

3. How effective is active learning in limiting the label budget to fine-tune foundational models?

# References

1. Ash, Jordan T., et al. "Deep batch active learning by diverse, uncertain gradient lower bounds." arXiv preprint arXiv:1906.03671 (2019).

2. Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations, 2018.

3. Dan Wang and Yi Shang. A new active labeling method for deep learning. In 2014 International joint conference on neural networks, 2014.

4. Dan Roth and Kevin Small. Margin-based active learning for structured output spaces. In European Conference on Machine Learning, 2006.

5. Wei-Ning Hsu and Hsuan-Tien Lin. Active learning by learning. In Association for the advancement of artificial intelligence, 2015.

6. BADGE repo: https://github.com/JordanAsh/badge
