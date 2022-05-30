---
header-includes: |
	\usepackage{amsmath}
	\usepackage{fancyhdr}
	\usepackage{physics}
	\usepackage{hyperref}
	\usepackage{graphicx}
	\graphicspath{ {./images/} }
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
    \newcommand{\mle}[1]{\hat{#1}_{\text{MLE}}}
title:
- Regression II
author:
- COMP9417, 22T2
---

# Stats, stats, stats \ldots

## Probability Distribution

A probability distribution represents the probability we see a value $x$ in a sample $X$. We denote this as $P(X=x)$.

### Definition
- Probability *mass* function applies to discrete $X$
- Probability *density* function applies to continous $X$

---

## Expected Values
An expected value (denoted $\mathbb{E}$) represents the weighted average of the probability distribution. This is typically seen as the value the distribution will converge to over time if sampled randomly.

For a discrete random variable, the form for an expected value is as follows:
\begin{align*}
    \mathbb{E}(X) = \sum_{x \in X} x P(X = x)
\end{align*}

In the continuous case, where $f(x)$ is the probability density function:
\begin{align*}
    \mathbb{E}(X) = \int_{-\infty}^{\infty} x f(x) dx
\end{align*}


---

### Example
**Problem:** Model the probability mass function and find the expected value of the roll of a dice.

- $P(X = 1) = \frac{1}{6}$, $P(X = 2) = \frac{1}{6}$, \ldots, $P(X = 6) = \frac{1}{6}$

\vspace{12pt}
For the expected value:

\begin{align*}
\mathbb{E}(X) = 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + \cdots + 6 \cdot \frac{1}{6} = 3.5
\end{align*}

This means that if we roll a dice, overtime the average dice value will converge to $3.5$. 

---

\begin{center}
\includegraphics[scale=0.25]{tut2_diceroll.png}
\end{center}

---


We typically assume our samples are i.i.d (independent and identically distributed), helping us reduce the complexity of the problem and apply statistically supported conclusions.

## Gaussian Distribution

A standard probability distribution is the Gaussian, where:

:::: columns
::: column
\begin{align*}
   \theta &= (\mu, \sigma^2), \quad \mu \in \mathbb{R}, \sigma > 0 \\
   p_{\theta}(x) &= \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(x-\mu)^2}{2\sigma^2} \right)
\end{align*}

We typically write $X \sim \mathcal{N}(\mu, \sigma^2)$ to say $X$ is normally distributed with mean $\mu$ and variance $\sigma^2$.
:::
::: column

\begin{center}
\includegraphics[scale=0.6]{tut2_normal.png}
\end{center}

:::
::::


## Maximum Likelihood Estimation

Maximum likelihood estimation is the process of estimating the parameters of a distribution of sample data by maximising the overall likelihood of the samples occuring in the distribution.

\begin{align*}
    \text{Prob of observing } X_1, \ldots, X_n &= \text{Prob of observing } X_1 \times \cdots \times \text{Prob of observing } X_n \\
    &= p_\theta (X_1) \times \cdots \times p_\theta(X_n) \\
    &= \prod_{i=1}^n p_\theta (X_i) \\
    &=: L(\theta) \text{ \quad \quad this is our likelihood function}
\end{align*}

---

To make life easier, we typically work with the log of the likelhood function (log-likelihood). As $\log$ is a strictly increasing function, the maximisation of $L(\theta)$ and $\log(L(\theta))$ give us the same result.

\begin{align*}
    L(\theta) &= \prod_{i=1}^n p_\theta (X_i) \\
    \log(L(\theta)) &= \log \prod_{i=1}^n p_\theta (X_i) \\
    &= \sum_{i=1}^n \log p_\theta (X_i) \\
\end{align*}

This makes differentiating, and therefore maximising much easier.

# 1 (a, b)
## 1a

**Problem**: Given $X_1, \cdots, X_n \sim N(\mu, 1)$, find $\hat{\mu}_{\text{MLE}}$.

First, we define our likelihood function:

\begin{align*}
    \log L(\mu) &= \log \left( \prod_{i=1}^n p_\theta (X_i) \right) \\
    &= \log \left( \prod_{i=1}^n \frac{1}{\sqrt{2\pi}} \exp \left( -\frac{1}{2} (X_i - \mu)^2  \right)\right) \\
    &= -\frac{1}{2} \log(2\pi)  -\frac{1}{2} \sum_{i=1}^n (X_i - \mu)^2 \\
\end{align*}

---

Next, we differentiate with respect to our parameter $\mu$,

\vspace{-.5cm}
\begin{align*}
    \frac{\partial\log L(\mu)}{\partial\mu} &= \sum_{i=1}^n (X_i - \mu) \\
    &= \sum_{i=1}^n X_i - n\mu \\
    \frac{\partial\log L(\mu)}{\partial\hat{\mu}} = 0 & \text{ at the minimum. So, } \\
    \sum_{i=1}^n X_i - n \hat{\mu} &= 0 \\
    \hat{\mu} &= \frac{1}{n} \sum_{i=1}^n X_i \\
    \hat{\mu} &= \bar{X}
\end{align*}

## 1b

**Problem**: Given $X_1, \ldots, X_n \sim \text{Bernoulli}(p)$, find $\hat{p}_{\text{MLE}}$.

The Bernoulli distribution models processes with 2 outcomes (eg. a coin toss).

\vspace{-.5cm}

\begin{align*}
  P(X = k) &= p^{k}(1-p)^{1-k}, \quad\quad k=0,1 \quad p \in [0,1]
\end{align*}

First, we define our likelihood function:

\vspace{-.5cm}
\begin{align*}
    \log L(\mu) &= \log \left( \prod_{i=1}^n p^{X_i}(1-p)^{X_{i}} \right) \\
    &= \sum_{i=1}^{n} \log p^{X_i} + \sum_{i=1}^{n} \log (1-p)^{X_{i}} \\
    &= n \bar{X} \log p + n(1-\bar{X})\log(1-p)
\end{align*}

---

Next, we differentiate with respect to our parameter $p$,

\vspace{-.5cm}
\begin{align*}
    \frac{\partial\log L(p)}{\partial p} &= \frac{n \bar{X}}{p} - \frac{n(1-\bar{X})}{1-p} \\
    \frac{\partial\log L(p)}{\partial\hat{p}} = 0 & \text{ at the minimum. So, } \\
    \frac{n \bar{X}}{\hat{p}} - \frac{n(1-\bar{X})}{1-\hat{p}} &= 0 \\
    n \bar{X} - n \bar{X}\hat{p} &= n(1-\bar{X})\hat{p} \\
     \hat{p}(n(1-\bar{X}) + n \bar{X}) &=n \bar{X}  \\
     \hat{p} &= \bar{X}  \\
\end{align*}

# Bias \& Variance

## Bias
The bias of an estimator represents its theoretical error. This theoretical error is the distance of the expected value of the predicted estimator away from the true parameter. We take the expected value for a representation of the estimate over an infinitely large dataset.

In the case of a model, this represents the error of the model on the training set.

*There will be a more in-depth discussion of this later on in the course.*

\begin{align*}
  \text{bias}(\hat{\theta}) = \mathbb{E}(\hat{\theta}) - \theta
\end{align*}

## Variance

The variance of an estimator refers to how different or *variable* it is over different data. Again, we take the expected value to find the converging value over a theoretically infinitely large dataset.

When applying this to a model, we'll discover that a large variance and a low bias typically means that our model has *overfit* the training set.

\begin{align*}
  \text{var}(\hat{\theta}) &= \mathbb{E}(\theta - \mathbb{E}(\hat{\theta}))^{2} \\
\end{align*}

# 2 (a, b, c)
## 2a
*Problem*: Find the bias and variance of $\mle{\mu}$ for $\mathbf{X} \sim N(\mu, 1)$.

We know that $\mle{\mu} = \bar{X}$. So,
\begin{align*}
  \text{bias}(\bar{X}) &= \mathbb{E}(\bar{X}) - \mu \\
  &= \frac{1}{n} \mathbb{E}\left(\sum_{i=0}^{n}X_{i}\right) - \mu \\
  &= \frac{1}{n} \sum_{i=0}^{n}\mathbb{E}(X_{i}) - \mu \\
  &= \frac{1}{n} n\mu - \mu \\
  &= 0
\end{align*}

---

\begin{align*}
  \text{var}(\hat{\mu}) &= \text{var}(\bar{X}) \\
  &= \text{var}\left(\frac{1}{n} \sum_{i=0}^{n}\mathbb{E}(X_{i}) \right) \\
  &= \frac{1}{n^{2}} \sum_{i=0}^{n}\text{var}(\mathbb{E}(X_{i})) \\
  &= \frac{1}{n^{2}} n = \frac{1}{n}
\end{align*}


## 2b
## 2c
