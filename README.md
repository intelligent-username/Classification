# Classification

![Agnus Dei by Francisco de Zurbarán, 1635](imgs/cover.jpg)

In classification, we have some information and we want to *label* it as part of some discrete class.

## Introduction

Classification is sometimes called categorical regression (or logistic regression, if we're only categorizing between binary values), but it's not *really* regression. Regression refers to predicting continuous values. There's also no closed-form solution. To build intuition, we can think of classification tasks as regression with a transformed floor or ceiling function applied to the continuous prediction.

There are many different branches of classification. In this writeup, we'll cover some of the most important ones.

---

## Logistic Regression

Logistic regression is the simplest and most important form of classification to understand. It is a special case of classification that only has binary outputs. When performing logistic regression, we aim to find the best-fitting model to describe the relationship between the binary outcome and the predictor variables.

Just like linear regression, we have a set of weights $w$ which we try to optimize. With inputs $x$ and outputs $y$, we try to learn what parameters $w$ will best make future predictions $\hat{y}$. Our goal is to minimize the loss, that way we have the most accurate model possible. Of course, there are other considerations, like overfitting, but those will be covered later.

### Logistic Function (Sigmoid)

Unlike in linear regression, we don't just simply output the weighted + biased sum of the inputs. Instead, we map our intermediate result $z$, which is defined as:

$$
z = w \cdot x = w_0 + w_1 x_1 + w_2 x_2 + ... + w_n x_n
$$

and then apply the logistic (sigmoid) function to it:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

which will give us an output between 0 and 1. We interpret this as the probability that the input belongs to the *positive* class (label 1). Then, we set some threshold for our predictions. For example, only if something has a 70% chance of being spam do we actually classify it as spam.

![Sigmoid Function](imgs/sigmoid.png)

### Negative Log-Likelihood

$$
L(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

Just like other forms of regression, we need a measure of our accuracy to know how good our model is. In logistic regression, the loss function we use is [negative log-likelihood function](https://github.com/intelligent-username/Loss-Functions?tab=readme-ov-file#7-negative-log-likelihood-nll) as our loss function. We try to minimize the average value of this function in order to improve our model. Since there is no closed-form solution, we are forced to use gradient descent in order to optimize the weights.

### Gradient Descent

We perform [gradient descent](https://github.com/intelligent-username/Gradient-Descent), just like in linear regression, to optimize our weights. The gradients of the loss function with respect to the weights are calculated as follows:

$$
\frac{\partial L}{\partial w_j} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i) x_{ij}
$$

We continuously update the weights using these gradients until convergence:

$$
w_{j+1} = w_j - \eta \left.\frac{\partial L}{\partial w}\right|_{w = w_j}
$$

As a reminder, our $L$ is the negative log-likelihood function, $N$ is the number of samples, $y_i$ is the true label for sample $i$, $\hat{y}_i$ is the predicted probability for sample $i$, and $x_{ij}$ is the value of feature $j$ for sample $i$.

We continue this descent until one of the convergence criteria is met.

---

## Decision Trees

![Decision Tree Illustration](imgs/DT.png)

Decision trees are to logistic regression what neural networks are to linear regression. They are a more complex and capture non-linear relationships between features and labels. However, instead of gradient descent, we use a greedy search algorithm, and instead of minimizing a loss function, we maximize *information gain*.

The main idea behind decision trees is that, we start with some data, and we try to segregate it consistently based on its features until we reach a point where we can confidently distinguish between classes. This is done by selecting the best features to split on at each node in the tree. This will result in a rooted tree where each leaf node represents a class label.

In each branching node of a graph, a feature must be examined to see whether its above or below some threshold, and classified accordingly. We continue this process recursively until we reach a stopping criterion, such as a maximum tree depth or a minimum number of samples per leaf.

### Splitting Criteria

This is the most important part to understand. Say we have some labelled data and we want to understand how to split them in a way such that examples of the same class are grouped together when we split on some feature.

If coming from a regression mindset, you might think of this as an iterative process, but it's not. Instead, we choose an impurity measure and run its corresponding, deterministic (and recursive) algorithm to find the best features and thresholds to split on. The technique we use will depend on the context, but the goal is to choose splits that most improve node purity according to the chosen criterion.

There are three main methods used in decision trees:

### Entropy & ID3

If our impurity measure is entropy, we can use the ID3 algorithm to build our decision tree.

The word 'entropy' (from the Greek 'ἐν' meaning 'in'and 'τροπή' meaning 'transformation') is used to describe the 'inner change' or disorder within a system. In the context of information theory, this is what we call our uncertainty about a random variable.

For a node containing $K$ classes, the entropy $H$ is:

$$
H(S) = -\sum_{i=1}^{K} p_i \log_2(p_i)
$$

Where:

- $S$ is the set of samples at the node.
- $p_i$ is the proportion of samples in class $i$ (a.k.a. the probability of class $i$ given random sampling).
- $K$ is the total number of classes.

In other words, our entropy measures the *opposite* of how confident we are in our predictions. The goal, of course, is to be as confident as possible. So, we must minimize entropy.

To do this, we use the concept of *information gain*. Information gain is how much our confidence increases (how much entropy decreases) after a certain operation is performed. The ID3 algorithm works with this concept to recursively build our decision trees. Its steps are:

1. Starting at the current root node, calculate the entropy.
2. Look at all the candidate features and split the dataset on each feature.
3. For each split, calculate the *new* expected entropy, using the same formula as step 1, this time finding the weighted average:

    $$
    H_{new} = \sum_{j=1}^{M} \frac{|S_j|}{|S|} H(S_j)
    $$

    ($M$ is the number of splits, $S_j$ is the set of samples in split $j$, and $|S|$ is the total number of samples at the current node)

4. Choose the split that results in the highest information gain (i.e., with the lowest entropy):

    $$
        \text{IG} := \text{Information Gain} = H(S) - H_{j}
    $$
    (where $j$ is each possible split (at a given depth) and $\text{IG}$ is the list of information gains.)
    $$
        \text{NS} := \text{NewSplit} = \max \text{IG}
    $$

5. Now, each split will create child nodes. Recurse by making each child node the new root and repeating steps 1-5 until it's time to stop.

We stop when one of the following conditions is met:

- Entropy increases after all possible splits.
- All samples at a node belong to the same class.
- There are no remaining features to split on.
- The maximum tree depth is reached.
- There are fewer than a minimum number of samples at a node.

### Gain Ratio & C4.5

The idea behind using the Gain Ratio is that, oftentimes, when we try to minimize entropy, we will create many splits in order to micro-optimize the entropy. For example, we may get as minute as splitting on specific ID numbers. Of course, this can lead to overfitting. By penalizing a high number of splits, we make trees that generalize better.

The Gain Ratio is a modification of Information Gain method that takes into account the number of branches created by a split.

The Entropy + ID3 method by using Gain Ratio + C4.5 algorithm instead. Note that C4.5 is identical to ID3 (except for the purity measure it uses).

It's defined as:

$$
\text{Gain Ratio} = \frac{\text{Information Gain}}{\text{Split Information}}
$$

Where Information Gain is the same as earlier:

$$
\text{Information Gain} = H(S) - H_{j}
$$

And Split Information is the entropy of the split itself, calculated as:

$$
\text{Split Information} = -\sum_{j=1}^{M} \frac{||S_j||}{||S||} \log_2\left(\frac{||S_j||}{||S||}\right)
$$

Here, $||S_j||$ and $||S||$ represent the number of samples in split $j$ and the total number of samples at the current node, respectively.

We divide the number of splits by the total number of samples in order to find out what proportion of the samples fall within that split. Then, we multiply that proportion by the $\log_2$ of itself to find the final entropy of the split.

Notice how the logarithm of a small fraction is very negative, which will give us a very large (positive) split information values. In other words, when we have many small, evenly sized children we get high SplitInfo, and when we have one big child, and a few tiny splits we get low SplitInfo. Thus, the Gain Ratio will be lower for splits that create many small branches. Since we pick the split with the highest Gain Ratio, this new impurity measure will encourage balanced, binary splits.

Note that, despite these optimizations, we often still stick to Entropy + ID3 in practice because it's simpler

### Gini Impurity & CART

The idea behihnd Gini Impurity + Cart is to split the data in binary splits every time.

Gini impurity can be written as:

$$
G(S) = 1 - \sum_{i=1}^{K} p_i^2
$$

Now, since we split into 'left' and 'right' branches, to get the final impurity of a split, we take the weighted average of the Gini impurities of the two branches:

$$
G_{split} = \frac{||S_{left}||}{||S||} G(S_{left}) + \frac{||S_{right}||}{||S||} G(S_{right})
$$

(Where $S_{left}$ and $S_{right}$ are the sets of samples in the left and right branches, respectively.)

Now, the CART (Classification and Regression Trees) algorithm works similarly to ID3 and C4.5, checking every possible split and picking the one with the least impurity:

$$
NS := NewSplit = \min G_{split}
$$

### Making Predictions

Once we choose an impurity measure and run the corresponding greedy algorithm, we can make predictions by traversing the tree from the root to a leaf node based on the feature values of the input data point. The label associated with the leaf becomes our predicted label.

### Limitations

- Can overfit
- Use greedy algorithms, meaning they may sacrifice global optimality for instant gratification. As a result, they also can't be optimized directly, pruned, or regularized.
- Too flat: can't model complex curves

To solve these issues, we can use gradient boosting and random forests. But notice that, already, we're no longer limited to binary decisions.

---

## Support Vector Machines

---

## Multi-Class Classification

---

## k-Nearest Neighbours

It would be a shame to talk about classification methods without at least mentioning k-nearest neighbours. It's not very closely related to any of the other methods we've discussed, but it's so effective that one cannot ignore it.

When using k-NN, we take a point's features and look around at the k closest points in our training data. We then take a majority vote of those k points' labels to determine the label of our input point. it is a very simple algorithm, but it often works very well. Also, k-NN gives us an early and simplistic preview into unsupervised learning.

---

## Setup

---

This project is licensed under the [MIT License](LICENSE).
