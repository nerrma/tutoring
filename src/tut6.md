---
header-includes: |
	\usepackage{amsmath}
	\usepackage{fancyhdr}
	\usepackage{physics}
	\usepackage{hyperref}
	\usepackage{pgfplots}
	\usepackage{algpseudocode}
	\usepackage{graphicx}
	\graphicspath{ {./images/} }
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
title:
- Ensemble Methods
author:
- COMP9417, 22T2
---

# Ensemble Methods

Arguably the most powerful non *deep-learning* methods, coming close to the performance of neural networks and still winning Kaggle competitions.

\pause

**Why?**

# Quick Recap: Bias and Variance of Estimators

Recall the bias of an estimator $\hat{\theta}$ is defined as:
\begin{align*}
  \text{bias}(\hat{\theta}) = \mathbb{E}(\hat{\theta}) - \theta
\end{align*}

And its variance is defined as:
\begin{align*}
  \text{var}(\hat{\theta}) &= \mathbb{E}\left[(\theta - \mathbb{E}[\hat{\theta}])^{2}\right] \\
\end{align*}

---


\centering
\includegraphics[scale=0.25]{tut6_bvar_shots.png}


# Bias-Variance Tradeoff

Recall the bias-variance decomposition of the MSE for an estimator $\hat{\theta}$:

\begin{align*}
  \text{MSE}(\hat{\theta}) =  \text{var}(\hat{\theta}) + \text{bias}(\hat{\theta})^{2}
\end{align*}

obviously for the best estimator we need to minimise the variance and minimise the bias.

---

**However**, if we try and minimise the bias, we typically also increase variance.

\pause

\centering
\includegraphics[scale=0.5]{tut6_bvar_esl.png}

# Bagging

Bagging or **B**oostrap **Agg**regation is an ensemble method we can apply to reduce the variance of our model.

\pause

We typically take models which are easy to train and suffer from high variance (i.e decision trees), fit their basic forms on different parts of our dataset and aggregate them into a *committee*.

\pause

For example, if we have a dataset $D = (x_{i}, y_{i})$ for $i \in [1, n]$, we might train 4 decision trees on $m$ points (where $m = n/4$) randomly picked from our dataset. \pause We then have a committee of four trees with distinct knowledge on the dataset, which we can then average for our final prediction.

---

Generally, if we take $B$ separate training sets from data $D$, our bootstrapped models will be:

\begin{align*}
  \hat{f}^{1}(D_{1}), \hat{f}^{2}(D_{2}), \ldots, \hat{f}^{B}(D_{B})
\end{align*}
\pause

and the final prediction for a point $x$ is
\begin{align*}
  \hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B}\hat{f}^{b}(x)
\end{align*}

---

**Why does this work in reducing variance?**

\pause

If we consider a statistical learning problem, where we have iid. data $X_{1}, \ldots, X_{n} \sim N(\mu, \sigma^{2})$ and we try finding an estimator $\hat{\mu}$ for the mean $\mu$. \pause

Consider an averaging estimator, where
\begin{align*}
  \only<3>{\hat{\mu}= \frac{1}{n} \sum_{i=1}^{n} X_{i}}
  \only<4>{\mathbb{E}\left(\frac{1}{n} \sum_{i=1}^{n} X_{i}\right) &= \frac{1}{n}\mathbb{E}\left(\sum_{i=1}^{n} X_{i}\right)}
  \only<5->{\mathbb{E}\left(\frac{1}{n} \sum_{i=1}^{n} X_{i}\right) &= \mu} \\
  \only<6>{\text{var}\left(\frac{1}{n} \sum_{i=1}^{n} X_{i}\right) &= \frac{1}{n^{2}} \text{var}\left(\sum_{i=1}^{n} X_{i}\right)}
  \only<7->{\text{var}\left(\frac{1}{n} \sum_{i=1}^{n} X_{i}\right) &= \frac{\sigma^{2}}{n}}
\end{align*}

# Random Forests

In bootstrap aggregation, the trees we generate may be *correlated*. To combat this we introduce *random forests* where:

- Randomly pick bootstrap samples \pause
- At every step of tree learning, randomise what features the tree splits on

	- Typically we pick $m \approx \sqrt{p}$ features for the trees to split on

\pause

Rationale: if we have strong predictors/features in our dataset, bagged trees will all typically pick the same features, leading to highly correlated predictions within the committee. This methods reduces this correlation and therefore the variance.

# Boosting

In boosting, we use a weak learner and improve it incrementally by adding more weak learners to make up for its mistakes. \pause So we'll have a final model in the form,

\begin{align*}
  C_{m}(X) = \alpha_{1} h_{1}(X) + \alpha_{2} h_{2}(X) + \ldots + \alpha_{m} h_{m}(X)
\end{align*}

where $\alpha_{i}$ signifies the influence/weighting we give a model $h_{i}$ for the final decision. \pause

We also define a $w_{i}$ for each iteration, signifying the weighting of each point. As each subsequent model needs to be an improvement on the last, we use these weights to signify which point the previous model misclassified.

---

\centering
\includegraphics[scale=0.5]{tut6_comparison.png}

## Adaboost

Let's take a look at the **Ada**ptive **Boost**ing algorithm. \pause

For a binary classification problem, we'll define the exponential loss as:

\begin{align*}
  L(h(x_{i}), y_{i}) = e^{-y_{i} h(x_{i})}
\end{align*}

this loss typically isn't used in practice, but gives us a way of *weighting* how good a model performs on a dataset.

---

Recall, our boosted model takes the form: $C_{m}(X) = \alpha_{1} h_{1}(X) + \alpha_{2} h_{2}(X) + \ldots + \alpha_{m} h_{m}(X)$.

So, our total loss will be:

\begin{align*}
  \only<1->{L(C_{m}(X), Y) &= \sum_{i=1}^{n} e^{-y_{i} C_{m}(x_{i})} \\}
  \only<2>{&= \sum_{i=1}^{n} e^{-y_{i} C_{m}(x_{i})} \\}
  \only<3->{&= \sum_{i=1}^{n} e^{-y_{i} (C_{m-1}(x_{i}) + \alpha_{m} h_{m}(x_{i}))} \\}
  \only<4->{&= \sum_{i=1}^{n} e^{-y_{i} C_{m-1}(x_{i})} e^{-y_{i}\alpha_{m} h_{m}(x_{i})} \\}
  \only<5->{&= \sum_{i=1}^{n} w_{i}^{m} e^{-y_{i}\alpha_{m} h_{m}(x_{i})} \\}
\end{align*}


---

\begin{align*}
  \only<1>{L(C_{m}(X), Y)&= \sum_{i=1}^{n} w_{i}^{m} e^{-y_{i}\alpha_{m} h_{m}(x_{i})} \\}
  \only<2->{L(C_{m}(X), Y)&= \sum_{y_{i} = h_{m}(x_{i})} w_{i}^{m} e^{-\alpha_{m}} + \sum_{y_{i} \neq h_{m}(x_{i})} w_{i}^{m} e^{\alpha_{m} } \\}
\end{align*}

\pause
\pause

So, our problem is essentially,
\begin{align*}
  \argmin_{\alpha, h} \left( \sum_{y_{i} = h_{m}(x_{i})} w_{i}^{m} e^{-\alpha_{m}} + \sum_{y_{i} \neq h_{m}(x_{i})} w_{i}^{m} e^{\alpha_{m} } \right)
\end{align*}

---

\begin{align*}
  \frac{\partial L}{\partial \alpha} = -e^{-\alpha_{m}} \sum_{y_{i} = h_{m}(x_{i})} w_{i}^{m}  +  e^{\alpha_{m} } \sum_{y_{i} \neq h_{m}(x_{i})} w_{i}^{m}
\end{align*}

\pause
At the minimum:

\begin{align*}
  \only<2>{-e^{-\alpha_{m}} \sum_{y_{i} = h_{m}(x_{i})} w_{i}^{m}  +  e^{\alpha_{m} } \sum_{y_{i} \neq h_{m}(x_{i})} w_{i}^{m} = 0} \\
  \only<3->{e^{2\alpha_{m}} &= \frac{\sum_{y_{i} = h_{m}(x_{i})} w_{i}^{m}}{\sum_{y_{i} \neq h_{m}(x_{i})} w_{i}^{m}}} \\
  \only<4->{\alpha_{m} &= \frac{1}{2} \log \left( \frac{\sum_{y_{i} = h_{m}(x_{i})} w_{i}^{m}}{\sum_{y_{i} \neq h_{m}(x_{i})} w_{i}^{m}} \right)} \\
\end{align*}

---

If we let

\begin{align*}
  \epsilon_{m} = \frac{\sum_{y_{i} \neq h_{m}(x_{i})} w_{i}^{m}}{\sum_{i=1}^{n} w_{i}^{m}}
\end{align*}

We can redefine $\alpha_{m}$ as:

\begin{align*}
 \alpha_{m} = \frac{1}{2} \log \left(\frac{1-\epsilon_{m}}{\epsilon_{m}} \right)
\end{align*}

---

To actually get a form for $w_{i}^{(m)}$, we can apply the same trick of recursion,

\begin{align*}
  w_{i}^{(m)} &= e^{-y_{i} C_{m-1}(x_{i})} \\
  &= e^{-y_{i} (C_{m-2}(x_{i}) + \alpha_{m-1} h_{m-1}(x_{i}))} \\
  &= w_{i}^{(m-1)} e^{-y_{i} \alpha_{m-1} h_{m-1}(x_{i})} \\
\end{align*}
\pause
So, when $y_{i} = h_{m-1}(x_{i})$:
\begin{align*}
  w_{i}^{(m)} &= w_{i}^{(m-1)} e^{-\alpha_{m-1}} \\
\end{align*}
\pause
When $y_{i} \neq h_{m-1}(x_{i})$:
\begin{align*}
  w_{i}^{(m)} &= w_{i}^{(m-1)} e^{\alpha_{m-1}} \\
\end{align*}

---

Now we have our definitions, we can define the **Adaboost** algorithm.

If we have a dataset $D = (X, y)$ where $X \in \mathbb{R}^{n \times p}$ and $y \in \mathbb{R}^{n}$. Where $T$ is our ensemble size and we have a learning algorithm $A$.
\begin{algorithmic}
  \State $w^{(1)} \gets \frac{1}{n}$
  \For{$t = 1, \ldots, T$}
  \State $M_{t} \gets A(X, w^{(t)})$
  \State $\alpha_{t} \gets \frac{1}{2} \log \left(\frac{1-\epsilon_{t}}{\epsilon_{t}} \right)$
  \State $w_{i}^{(t+1)} \gets w_j^{(t)} \exp(\alpha_{t}) \qquad j \text{ where } y_{j} \neq M_{t}(x_{j})$
  \State $w_{j}^{(t+1)} \gets w_j^{(t)} \exp(-\alpha_{t}) \qquad j \text{ where } y_{j} = M_{t}(x_{j})$
  \EndFor
  \Return $M(X) =\text{sgn}\left(\sum_{t=1}^{T} \alpha_{t} M_{t}(X)\right)$
\end{algorithmic}
