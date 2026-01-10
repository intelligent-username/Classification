# Classification

![Agnus Dei by Francisco de Zurbarán, 1635](imgs/cover.jpg)

In classification, we have some information and we want to *label* it as part of some discrete class.

## Outline

- [Introduction](#introduction)
- [Logistic Regression](#logistic-regression)
  - [Logistic Function (Sigmoid)](#logistic-function-sigmoid)
  - [Negative Log-Likelihood](#negative-log-likelihood)
  - [Gradient Descent](#gradient-descent)

- [Decision Trees](#decision-trees)
  - [Splitting Criteria](#splitting-criteria)
  - [Entropy & ID3](#entropy--id3)
  - [Gain Ratio & C4.5](#gain-ratio--c45)
  - [Gini Impurity & CART](#gini-impurity--cart)
  - [Making Predictions](#making-predictions)
  - [Limitations](#limitations)

- [Gradient-Boosted Trees](#gradient-boosted-trees)
  - [Process](#process)
  - [How to do step 3](#how-to-do-step-3)
  - [Key Properties](#key-properties)

- [XGBoost](#xgboost)

- [Random Forests](#random-forests)

- [Support Vector Machine](#support-vector-machine)

- [Multiclass Classification](#multiclass-classification)
  - [Strategies for Multiclass SVMs](#strategies-for-multiclass-svms)
  - [Multiclass Logistic Regression](#multiclass-logistic-regression)
  - [Decision Trees and Ensembles](#decision-trees-and-ensembles)

- [k-Nearest Neighbours](#k-nearest-neighbours)

## Introduction

Classification is sometimes called categorical regression (or logistic regression, if we're only categorizing between binary values), but it's not *really* regression. Regression refers to predicting continuous values. Classification is more like "threshold prediction", i.e. performing classification is like performing regression and then applying a custom floor/ceiling function to the predicted values.

There are many different branches of classification. In this writeup, we'll cover the most important ones.

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

which will give us an output between 0 and 1. We interpret this as the probability that the input belongs to the *positive* class (label 1). Then, we set some threshold for our predictions. For example, we can say, "only if something has a chance of at least 70% of being spam will we classify it as spam".

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

As a reminder:

- $L$ is the negative log-likelihood function
- $N$ is the number of samples, $y_i$ is the true label for sample $i$
- $\hat{y}_i$ is the predicted probability for sample $i$
- $x_{ij}$ is the value of feature $j$ for sample $i$.

We continue this descent until one of the convergence criteria is met.

So, in summation, when performing logistic regression, we initialize some set of weights $w$, add a sigmoid (activation) layer to the end of the model, and then use gradient descent to optimize the weights according to the negative log-likelihood loss function. The output of these weights is then pass into a "classifier", in this case the sigmoid function, which maps the output to a probability between 0 and 1. We can change the threshold for a 'positive' classification depending on the tradeoff we want to make between precision and recall.

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

   $M$ is the number of splits, $S_j$ is the set of samples in split $j$, and $|S|$ is the total number of samples at the current node

4. Choose the split that results in the highest information gain (i.e., with the lowest entropy):

   $$
   \text{IG} := \text{Information Gain} = H(S) - H_{j}
   $$

   Where $j$ is each possible split (at a given depth) and $\text{IG}$ is the list of information gains.

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

Once we choose an impurity measure and run the greedy algorithm, an input is labelled by looking at its features and using them to traverse down the tree from root to leaf. Once we reach a leaf, we assign the most common class that is found within that leaf to our input point.

### Limitations

- Can overfit
- Greedy algorithms often sacrifice global optimality for instant gratification.
- Too flat: can't model complex curves

To solve these issues, we can use gradient boosting and random forests. But notice that, already, we're no longer limited to binary decisions.

## Gradient-Boosted Trees

![Gradient Boosting](imgs/GBT.png)

Gradient-boosted trees take the basic decision tree idea and turn it into a **sequence of weak learners** that correct each other’s mistakes. Using them is like performing gradient descent on decision trees, with which we continue itereating until the model converges.

Instead of building a single, deep tree, we build many small trees one after another. Each new tree looks at where the previous trees predicted incorrectly and tries to reduce that error.

### Process

1. **Start** with a single tree that predicts all outputs roughly. Often, this just means taking the mean of the target's values and making that the initial prediction.

    The prediction can be written as $\hat{y}^{(0)} = \frac{1}{N} \sum_{i=1}^{N} y_i$

    (Note that the 0 is not exponentiation but the iteration index.)

    At this point in the iteration, we just have a tree with a single leaf.

    Note: if we are classifying, then there is no clear 'average', we instead use the log-odds of the positive class proportion, $\log\left(\frac{p}{1-p}\right)$.

    So, in other words, when we're predicting some continuous value, we always guess the mean, and when we're classifying, we always predict that the input has $log\left(\frac{p}{1-p}\right)$ chance of being in the positive class.

2. **Compute the residuals** (prediction vs. ground truth for every sample).

    The residual for each sample can be calculated as:

   $$
   r^{(j)} = y^{(j)} - \hat{y}^{(j)}
   $$

   Where $j$ is the current iteration of the boosting process and $y$ is the true label.

   This calculation is done for every sample. Notice that we don't square or take the absolute value of the errors, instead we want to keep track of how much each prediction strayed from the truth.

3. Train a new tree to predict *these* residuals. We train on the residuals in order to understand what we're doing wrong in the current ensemble (group) of trees and move in the opposite direction.

4. Add this tree’s predictions to the previous ones, scaled by a learning rate $\eta$:

   $$
   \hat{y}^{(j+1)} = \hat{y}^{(j)} + \eta \cdot \text{tree's output}
   $$

      You can think of this as performing gradient descent on the predictions, where we have:

   $$
   y^{(\text{new})}_j = y^{(\text{old})}_j - \eta \cdot \frac{\partial L}{\partial y_j} \\
   = y^{(\text{old})}_j - \eta (\hat{y}^{(j)} - y^{(j)}) \\
   = y^{(\text{old})}_j - \eta \cdot (-r^{(j)}) \\
   = y^{(\text{old})}_j + \eta \cdot r^{(j)}
   $$

      Note that the new indices of $y$ and $\hat{y}$ don't indicate an *updated prediction* by the same tree, but rather the set of predictions by the new, updated tree (the latest in the ensemble).

5. Recurse steps 2 to 4 until convergence.

Each tree will only focus on the mistakes of the ensemble so far. We adjust in the direction of the negative gradient and scale that adjustment by the learning rate, that way we "converge slower" to create a more complex tree without overfitting. This process gradually improves the model. The small size of each tree and the learning rate prevent overfitting, even when we build hundreds of trees.

### How to do step 3

At step 3, we have our input features as well as a 'residual' value for each sample. Here, we're working with continuous values (doing regression) since we're predicting residuals. Once again, we use a greedy algorithm to build a decision tree.

For example, if using the Gain Ratio + C4.5 method, we would:

1. Find all possible splits to split on for each feature.

2. Find the split that minimizes the variance of the residuals in each child node. (Since we're doing regression now, we want to minimize variance instead of entropy.)
3. Create child nodes and recurse until a stopping criterion is met.

### Key Properties

- Trees are shallow (often 3–8 levels) and additive.
- The same impurity measures (Gini, entropy) to build each tree.
- Trees are invariant to monotonic transformations (don't need normalization).
- We can now use trees to model more complex, non-linear data.
- Work well with smaller datasets.

## XGBoost

Extreme Gradient Boosting is an implementation of Gradient-Boosted trees that's implemented in C++ and optimized for speed and performance. It includes several enhancements over basic gradient boosting, such as:

- Regularization: XGBoost includes L1 and L2 regularization to prevent overfitting.
- Automatically handles missing values during training
- Tree pruning (via `max_depth` and `min_child_weight`) and parallelization for faster training.
- Is easier to analyze since it contains a standardized interface and features.

In practice, when we want to make a gradient-boosted tree, we just import the XGBoost library and use its built-in functions to create and train our model. It abstracts away the complexities of implementing gradient boosting from scratch, allowing us to focus on tuning hyperparameters and improving model performance.

## Random Forests

![A Beautiful Drawing of an Abstract Random Forest](imgs/RF.png)

Building random forests is simpler than gradient boosting, but more complicated than basic decision trees. The main idea is, we have some data with some set of features. Then, we consider random subsets of these features a given of times and use a greedy build a decision tree for each. This creates an ensemble, which we use to make predictions.

Random forests build an ensemble in a different way: instead of sequentially fixing mistakes, they train **many independent decision trees** on random subsets of the data and features, then average their predictions.

### Steps

1. Start with the full dataset $S$.
2. If creating an $m$-tree ensemble with $n$ features each, trained on $k$ samples, repeat the following $m$ times:
   1. Take the dataset $S$ and randomly pick $n$ features that it will consider.
   2. Randomly take $k$ samples with replacement.
   3. Use a greedy algorithm to build a decision tree on this subset.
3. Now, you have $m$ k by n decision trees.

When using this ensemble, we pass the input features to each of the $m$ trees, collect their predictions, and then average them (for regression) or take a plurality vote (for classification) to get the final output.

Each tree is a weak, high-variance model, but combining them reduces variance by a lot. Random subsets of data and features ensure that the trees are decorrelated, so their errors tend to cancel out.

## Support Vector Machine

![SVM Illustration](imgs/SVM.png)

When working with a support vector machine (SVM), we try to find the best way to separate our classes. This means forming a line, 2D plane, or etc. to model a boundary. The hyperplane is formed through two **support vectors**. The best hyperplane is the one that maximizes the margin between the two classes. The margin is defined as the distance between the hyperplane and the closest points from each class, which are called support vectors.

To use an SVM, we simply plot the unlabeled data point and see which side of the hyperplane it's on. But first, we need to understand how the more basic versions of it work.

### Terminology

- In $\mathbb{R}^n$, a **hyperplane** is a $\mathbb{R}^{n-1}$ flat affine space. It can be written as:

$$
\beta_0 + \beta_1 x_2 + ... + \beta_n X_n = 0
$$

&nbsp; (Any point $X$ with coordinates $(X_1, X_2, ...\,, X_n)^T$ lies on the plane if it's components satisfy the above equation. Points that don't satisfy this equation lie on either "one side or the other" of the hyperplane. More on this later). For example, in $\mathbb{R}^2$, a hyperplane is just a line, and in $\mathbb{R}^3$, it's a 2D plane.

- The **margin** is the perpendicular distance between the hyperplane and the closest points from each class (the *support vectors*).

- $\text{exp(x)}$ is $e^{x}$

### Maximal margin Classifier

In general, when working with hyperplanes that correctly classify two classes, we can draw infinitely many hyperplanes that separate them. The Maximal Margin Classifier is the hyperplane that *maximizes* the margin between the two classes while still classifying all of the data points correctly.

In mathematical form, if we have a set of parameters ${\beta_1, ..., \beta_n}$, where each parameter is a set composed of weights $\beta_1, ... , \beta_n$, the maximal margin classifier solves the following optimization problem:

$$
\beta_{min} = \min_{\beta_i}(S), \text{where} \\
S =  \{\frac{1}{2} ||{\beta_i}|| ^2 : \beta_i \in \beta \}
$$

(note that we take min instead of the max since the margin, $\gamma$, is defined as $\frac{1}{||\beta||}$).

In other words, we try to reduce both bias and variance at the same time, which creates a nice, generalizable model.

### Support Vector Classifier

If there are no hyperplanes that can perfectly separate the two classes (or even if there are and we want to improve robustness), we need to further generalize the margin classifiers to fit our data.

A support vector classifier uses softer margins in order to allow for some misclassification. This introduces "slack variables" $\xi_i \ge 0$ to allow us more flexibility when fitting the hyperplane.:

$$
y_i (w \cdot x_i + b) \ge 1 - \xi_i, \quad \forall i
$$

and minimizes

$$
\frac{1}{2} \| \beta \|^2 + C \sum_i \xi_i
$$

where $C$ controls the trade-off between margin size and misclassification penalty.

Think of $\text{C}$ as our "variance budget", where, the higher our $\text{C}$, the more variance we're willing to accept in order to reduce bias (by fitting the training data better). Conversely, a lower $\text{C}$ means we're prioritizing a larger margin (lower variance) at the cost of some misclassifications (higher bias).

So, $\text{C}$ is now a regularization hyperparameter that we can tune to find the best balance for our specific dataset.

### Support Vector Machines

Now, imagine even that's not enough. What if the data is completely linearly inseparable? As in, what if some of the points fully surround other points of a different class?

In that case, the job is a lot harder. We can try to map the data onto a higher dimensional space so that it *is* separable.

To do this, we need some function, call it $\phi(x)$, that maps our data from $\mathbb{R}^n$ to $\mathbb{R}^m$, where $m > n$. Then, we can try to find a hyperplane in this new space.

For example, if we have two-dimensional points that are not linearly separable, we can map them to three dimensions using:

$$
\phi(x_1, x_2) = (x_1, x_2, x_1^2x_2^2)
$$

Now, to to find the optimal hyperplane in this new space, we can use the same techniques as before (maximal margin classifier or support vector classifier).

I.e. we want to find $\beta$ such that:

$$
\beta_{min} = \min_{\beta} \frac{1}{2} ||\beta||^2
$$

subject to the constraints

$$
y_i (\beta \cdot \phi(x_i) + b) \ge 1, \quad \forall i
$$

where, $y_i \in \{-1, +1\}$, and the support vectors are the points for which the inequality is tight ($y_i (\beta \cdot \phi(x_i) + b) = 1$). This is the same maximal margin optimization as in the original 2D space, just applied in the transformed 3D space.

### Kernel Trick

If we have a high-dimensional space and many points, these calculations can get very expensive. However, we can use a trick called the kernel trick to avoid explicitly computing the mapping $\phi(x)$.

The kernel trick allows us to compute the inner products in the high-dimensional space without explicitly mapping the points. We define a kernel function $K(x_i, x_j)$ that computes the inner product of the mapped points:

$$
K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)
$$

Common kernel functions include:

- **Linear Kernel**: $K(x_i, x_j) = x_i \cdot x_j$
- **Polynomial Kernel**: $K(x_i, x_j) = (x_i \cdot x_j + 1)^d$
- **Radial Basis Function (RBF) Kernel**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$

## Multiclass Classification

![Multi-Class Classification Illustration](imgs/MCC.png)

Multiclass classification is the extension of binary classification to handle **more than two classes**. Instead of simply predicting a yes/no label, we want to assign each input to one of $K > 2$ categories.

### Strategies for Multiclass SVMs

SVMs are inherently binary, but we can adapt them using one of two common strategies:

1. **One-vs-Rest (OvR) / One-vs-All (OvA)**  
   - Train $K$ separate binary classifiers, each distinguishing one class from all others.  
   - For class $k$, label all examples in class $k$ as $+1$ and all others as $-1$.  
   - At prediction time, compute the decision function for all $K$ classifiers and pick the class with the **highest margin** (i.e., the classifier most confident in a positive prediction).

2. **One-vs-One (OvO)**  
   - Train a binary classifier for **every pair of classes**, resulting in $K(K-1)/2$ classifiers.  
   - Each classifier distinguishes between two classes, ignoring the rest.  
   - At prediction, each classifier votes for one of its two classes. The class with the **most votes** is chosen as the prediction.

### Multiclass Logistic Regression

For logistic regression, multiclass prediction is typically handled using **softmax regression**:

$$
P(y=k \mid x) = \frac{\exp(w_k \cdot x)}{\sum_{j=1}^{K} \exp(w_j \cdot x)}
$$

- $w_k$ is the weight vector for class $k$.  
- The denominator ensures that the predicted probabilities sum to 1 over all classes.  
- The predicted class is:

$$
\hat{y} = \arg\max_{k} P(y=k \mid x)
$$

The loss function becomes **categorical cross-entropy**:

$$
L = -\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log P(y_i=k \mid x_i)
$$

where $y_{ik}$ is 1 if sample $i$ belongs to class $k$, 0 otherwise.

### Decision Trees and Ensembles

Decision trees, random forests, and gradient-boosted trees naturally handle multiple classes without modification. Each leaf simply stores the most frequent class among the training samples that reach it. During prediction, the path from root to leaf determines the predicted class.

## k-Nearest Neighbours

![k-NN Illustration](imgs/kNN.png)

It would be a shame to talk about classification methods without at least mentioning k-nearest neighbours. It's not very closely related to any of the other methods we've discussed, but it's so effective that one cannot ignore it.

When using k-NN, we take a point's features and look around at the k closest points in our training data. We then take a majority vote of those k points' labels to determine the label of our input point. it is a very simple algorithm, but it often works very well. Also, k-NN gives us an early and simplistic preview into unsupervised learning.

---

This project is licensed under the [MIT License](LICENSE).
