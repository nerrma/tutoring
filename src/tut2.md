---
header-includes: |
	\usepackage{amsmath}
	\usepackage{fancyhdr}
	\usepackage{physics}
	\usepackage{hyperref}
	\usepackage{pgfplots}
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
An expected value (denoted $\mathbb{E}$) represents the weighted average of the probability distribution. This is typically seen as the value the random variable will converge to over time if sampled randomly.

For a discrete random variable, the form for an expected value is as follows:
\begin{align*}
    \mathbb{E}(X) = \sum_{x \in X} x P(X = x)
\end{align*}

In the continuous case, where $f(x)$ is the probability density function:
\begin{align*}
    \mathbb{E}(X) = \int_{-\infty}^{\infty} x f(x) dx
\end{align*}

---

### General rules

For random variables $X, Y$ and a constant $c$

- $\displaystyle \mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y]$
- $\displaystyle \mathbb{E}[cX] = c\mathbb{E}[X]$

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
    \frac{\partial\log L(\mu)}{\partial\hat{\mu}} = 0 & \text{ at the maximum. So, } \\
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
    \log L(p) &= \log \left( \prod_{i=1}^n p^{X_i}(1-p)^{X_{i}} \right) \\
    &= \sum_{i=1}^{n} \log p^{X_i} + \sum_{i=1}^{n} \log (1-p)^{X_{i}} \\
    &= n \bar{X} \log p + n(1-\bar{X})\log(1-p)
\end{align*}

---

Next, we differentiate with respect to our parameter $p$,

\vspace{-.5cm}
\begin{align*}
    \frac{\partial\log L(p)}{\partial p} &= \frac{n \bar{X}}{p} - \frac{n(1-\bar{X})}{1-p} \\
    \frac{\partial\log L(p)}{\partial\hat{p}} = 0 & \text{ at the maximum. So, } \\
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
*Problem*: Find the bias and variance of $\mle{\mu}$ for $X \sim N(\mu, 1)$.

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

Find the bias and variance of $\mle{p}$ for $X \sim$ Bernoulli$(p)$.

We know that $\mle{p} = \overline{X}$, so

\begin{align*}
  bias(\mle{p}) &= \mathbb{E}(\mle{p}) - \mu \\
  &= \overline{X} - \mu \\
  &= 0 \\
\end{align*}

---

\begin{align*}
  var(\mle{p}) &= var(\bar{X}) \\
  &= var\left( \frac{1}{n} \sum_{i=1}^{n} x_{i} \right) \\
  &=\frac{1}{n^{2}} \sum_{i=1}^{n}  var\left( x_{i} \right) \\
 &=\frac{1}{n^{2}} np(1-p) \\
  &= \frac{p(1-p)}{n}\\
\end{align*}

## 2c

Perform bias-variance decomposition i.e prove that MSE$(\hat{\theta}) =$bias$(\hat{\theta})^{2}$ + var$(\hat{\theta})$.

\begin{align*}
  \text{MSE}(\hat{\theta}) &= \mathbb{E}\left[(\hat{\theta} - \theta)^{2}\right] \\
  &= \mathbb{E}\left[ \left( \hat{\theta} - \mathbb{E}(\hat{\theta}) + \mathbb{E}(\hat{\theta}) - \theta\right)^{2}\right] \\
  &= \mathbb{E}\left[(\hat{\theta} - \mathbb{E}(\hat{\theta}))^{2} + 2(\hat{\theta} - \mathbb{E}(\hat{\theta}))(\mathbb{E}(\hat{\theta}) - \theta) + (\mathbb{E}(\hat{\theta}) - \theta)^{2}\right] \\
  &= \mathbb{E}(\hat{\theta} - \mathbb{E}(\hat{\theta}))^{2} + 2\mathbb{E}[(\hat{\theta} - \mathbb{E}(\hat{\theta}))(\mathbb{E}(\hat{\theta}) - \theta)] + \mathbb{E}((\mathbb{E}(\hat{\theta}) - \theta)^{2}) \\
  &= \mathbb{E}(\hat{\theta} - \mathbb{E}(\hat{\theta}))^{2} + 2(\mathbb{E}(\hat{\theta}) - \mathbb{E}(\hat{\theta}))(\mathbb{E}(\hat{\theta}) - \mathbb{E}(\theta)) + \mathbb{E}((\mathbb{E}(\hat{\theta}) - \theta)^{2}) \\
\end{align*}

---

\begin{align*}
  &= \mathbb{E}(\hat{\theta} - \mathbb{E}(\hat{\theta}))^{2} + \mathbb{E}(\mathbb{E}(\hat{\theta}) - \theta)^{2} \\
  &= \mathbb{E}(\hat{\theta} - \mathbb{E}(\hat{\theta}))^{2} + (\mathbb{E}(\hat{\theta}) - \theta)^{2} \\
  &= \text{var}(\hat{\theta}) + \text{bias}(\hat{\theta})^{2}
\end{align*}

# 3
## 3a

Paraphrasing the problem, we assume that our data has a linear relationship. This means we have,

\begin{align*}
  y &= x^{T} \beta^{*} + \epsilon, & \epsilon \sim N(0, \sigma^{2})
\end{align*}

So,
\begin{align*}
  y|x \sim N(x^{T} \beta^{*}, \sigma^{2})
\end{align*}

---

For multiple $x_{i}$, we typically write $y|X \sim N(X^{T} \beta^{*}, \sigma^{2} I)$ for our input matrix $X$.

Our log likelhood will be,
\begin{align*}
  \log L(\beta) &= \log P(y|X, \beta) \\
  &= \log \left( \prod_{i=1}^{n} P(y_{i}|x_{i}, \beta) \right)\\
  &= \sum_{i=1}^{n} \log P(y_{i}|x_{i}, \beta) \\
  &= \sum_{i=1}^{n} \log \left( \frac{1}{\sqrt{2\pi \sigma^{2}}} \exp \left( -\frac{(y_{i} - x_{i}^{T} \beta)^{2}}{2\sigma^2}  \right) \right) \\
\end{align*}

---

\begin{align*}
  &= n\log(\frac{1}{\sqrt{2\pi \sigma^{2}}}) - \frac{1}{2\sigma^{2}} \sum_{i=1}^{n} (y_{i} - x_{i}^{T} \beta)^{2} \\
  &= -\frac{n}{2}\log(2\pi \sigma^{2}) - \frac{1}{2\sigma^{2}} \norm{y - X\beta}_{2}^{2}
\end{align*}

So, to find $\mle{\beta}$, we solve:

\vspace{-0.5cm}
\begin{align*}
  \mle{\beta} &= \argmax_{\beta} -\frac{n}{2}\log(2\pi \sigma^{2}) - \frac{1}{2\sigma^{2}} \norm{y - X\beta}_{2}^{2} \\
  &= \argmax_{\beta} - \frac{1}{2\sigma^{2}} \norm{y - X\beta}_{2}^{2} \\
  &= \argmin_{\beta} \norm{y - X\beta}_{2}^{2}
\end{align*}

---

This is just the least squares problem. So our solution is
\begin{align*}
  \mle{\beta} &= (X^{T} X)^{-1} X^{T} y \\
  &= \hat{\beta}_{\text{LS}}
\end{align*}

# Norms

We define the $p$-norm of a vector $x = (x_{1}, x_{2}, \ldots, x_{n})$ as:

\vspace{-0.5cm}
\begin{align*}
  \norm{x}_{p} = \left( \sum_{i=1}^{n} |x_{i}|^{p}\right)^{\frac{1}{p}}
\end{align*}

For a norm $g$ and vectors $x, y$, $g$ needs to satisfy the following conditions to be a valid norm,
\begin{enumerate}
        \item Triangle inequality. $g(x+y) \leq g(x) + g(y)$
        \item Absolute homogeneity. For a constant $c$, $g(cx) = |c| g(x)$
        \item Positive definiteness. The vector $0$ should have norm $0$.
\end{enumerate}


## Euclidean norm

:::: columns
::: column
We've already encountered the Euclidean or $\ell2$ norm as
\begin{align*}
  \norm{x}_{2} &= \sqrt{\sum_{i} x_{i}^{2}}
\end{align*}

If we have a vector $\beta = (\beta_{1}, \beta_{2})$ we can geometrically interpret the 2-norm as
\begin{align*}
  \norm{x}_{2} &= \sqrt{\beta_{1}^{2} + \beta_{2}^{2}}
\end{align*}

:::
::: column
\begin{figure}
\begin{tikzpicture}[pics/axis/.style={code={
  \draw[-] (-2.5,0) -- (2.5,0) node[anchor=north west] {$\beta_1$};
  \draw[-] (0,-2.5) -- (0,2.5) node[anchor=south east] {$\beta_2$};
  }},pics/norm penalty/.style={code={
  \draw (0,1) to[out=-45+45,in=135-45]
  (1,0) to[out=-135+45,in=45-45]
  (0,-1) to[out=135+45,in=-45-45]
  (-1,0) to[out=45+45,in=-135-45]  cycle;}}]
\path (0,0) pic{axis} (0,0) pic[blue,thick,looseness=1]{norm penalty=0};
\end{tikzpicture}
\end{figure}
:::
::::

## $\ell1$ norm

:::: columns
::: column
The $\ell1$ norm is defined as:
\begin{align*}
  \norm{x}_{1} &= \sum_{i} |x_{i}|
\end{align*}

Again, if we have a vector $\beta = (\beta_{1}, \beta_{2})$, the plot of the $\ell1$ norm is:

:::
::: column
\begin{figure}
\begin{tikzpicture}[pics/axis/.style={code={
  \draw[-] (-2.5,0) -- (2.5,0) node[anchor=north west] {$\beta_1$};
  \draw[-] (0,-2.5) -- (0,2.5) node[anchor=south east] {$\beta_2$};
  }},pics/norm penalty/.style={code={
  \draw (0,1) to[out=-45,in=135]
  (1,0) to[out=-135,in=45]
  (0,-1) to[out=135,in=-45]
  (-1,0) to[out=45,in=-135]  cycle;}}]
\path (0,0) pic{axis} (0,0) pic[blue,thick,looseness=1]{norm penalty=0};
\end{tikzpicture}
\end{figure}
:::
::::

## $\infty$ norm

:::: columns
::: column
The $\infty$ norm is defined as:
\begin{align*}
  \norm{x}_{\infty} &= \max_{i} |x_{i}|
\end{align*}

Again, if we have a vector $\beta = (\beta_{1}, \beta_{2})$, the plot of the $\infty$ norm is:

:::
::: column
\begin{figure}
\begin{tikzpicture}[pics/axis/.style={code={
  \draw[-] (-2.5,0) -- (2.5,0) node[anchor=north west] {$\beta_1$};
  \draw[-] (0,-2.5) -- (0,2.5) node[anchor=south east] {$\beta_2$};
  }},pics/norm penalty/.style={code={
  \draw (1.25,-1.25) -- (1.25,1.25) -- (-1.25, 1.25) -- (-1.25,-1.25) -- (1.25,-1.25);}}]
\path (0,0) pic{axis} (0,0) pic[blue,thick,looseness=1]{norm penalty=0};
\end{tikzpicture}
\end{figure}
:::
::::

## $0.5$ norm

:::: columns
::: column
The $0.5$ norm is defined as:
\begin{align*}
  \norm{x}_{0.5} &= \left( \sum_{i} \sqrt{x_{i}} \right)^{2}
\end{align*}

Let's look further into this result, as we could possibly be breaking the rules of norms.
:::
::: column
\begin{figure}
\begin{tikzpicture}[pics/axis/.style={code={
  \draw[-] (-2.5,0) -- (2.5,0) node[anchor=north west] {$\beta_1$};
  \draw[-] (0,-2.5) -- (0,2.5) node[anchor=south east] {$\beta_2$};
  }},pics/norm penalty/.style={code={
  \draw (0,1) to[out=-45-20,in=135+20]
  (1,0) to[out=-135-20,in=45+20]
  (0,-1) to[out=135-20,in=-45+20]
  (-1,0) to[out=45-20,in=-135+20]  cycle;}}]
\path (0,0) pic{axis} (0,0) pic[blue,thick,looseness=1]{norm penalty=0};
\end{tikzpicture}
\end{figure}
:::
::::

---

Take a point $x = (0, x_{1})$ and $y = (y_{1}, 0)$,
\begin{align*}
  \norm{x}_{0.5} &= x_{1} \\
  \norm{y}_{0.5} &= y_{1} \\
  \norm{x + y}_{0.5} &= (\sqrt{x_{1}} + \sqrt{y_{1}})^{2} = x_{1} + 2 \sqrt{x_{1}} \sqrt{y_{1}} + y_{1} \\
\end{align*}

So, $\norm{x + y}_{0.5} > \norm{x}_{0.5} + \norm{y}_{0.5}$ and the triangle inequality does not hold. Therefore, the $p=0.5$ is not a valid norm.

---

The ridge regression problem can be written as:

\begin{align*}
  \hat{\beta}_{\text{ridge}} = \argmin_{\beta} \{ \norm{y - X \beta}_{2}^{2} + \lambda \norm{\beta}_{2}^{2} \}
\end{align*}

We can interpret the term $\lambda \norm{\beta}_{2}^{2}$ in the minimisation as finding the $\beta$ with the minimum $2-$norm (multiplied by $\lambda$) while solving the least squares problem.

So, for an arbitrary $k$, we can redefine our problem as

\begin{align*}
  \hat{\beta}_{\text{ridge}} &= \argmin_{\beta} \{ \norm{y - X \beta}_{2}^{2} \} & \text{ where } \norm{\beta}_{2} \leq k
\end{align*}

---

For the non-regularised problem, we have

\begin{figure}
\begin{tikzpicture}[pics/axis/.style={code={
  \draw[-] (-2.5,0) -- (2.5,0) node[anchor=north west] {$\beta_1$};
  \draw[-] (0,-2.5) -- (0,2.5) node[anchor=south east] {$\beta_2$};
  }},pics/norm penalty/.style={code={
  \draw (0,0.25) to[out=-45+45,in=135-45]
  (0.25,0) to[out=-135+45,in=45-45]
  (0,-0.25) to[out=135+45,in=-45-45]
  (-0.25,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,0.5) to[out=-45+45,in=135-45]
  (0.5,0) to[out=-135+45,in=45-45]
  (0,-0.5) to[out=135+45,in=-45-45]
  (-0.5,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,0.75) to[out=-45+45,in=135-45]
  (0.75,0) to[out=-135+45,in=45-45]
  (0,-0.75) to[out=135+45,in=-45-45]
  (-0.75,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,1) to[out=-45+45,in=135-45]
  (1,0) to[out=-135+45,in=45-45]
  (0,-1) to[out=135+45,in=-45-45]
  (-1,0) to[out=45+45,in=-135-45]
  cycle;
}}]
\path (0,0) pic{axis} (2,2) pic[red,thick,looseness=1]{norm penalty=0};
\node at (2, 2) {\textbullet};
\node at (2.5, 2) {$\hat{\beta}_{\text{LS}}$};
\end{tikzpicture}
\end{figure}

---

For the ridge regression problem,

\begin{figure}
\begin{tikzpicture}[pics/axis/.style={code={
  \draw[-] (-2.5,0) -- (2.5,0) node[anchor=north west] {$\beta_1$};
  \draw[-] (0,-2.5) -- (0,2.5) node[anchor=south east] {$\beta_2$};
  }},pics/norm penalty/.style={code={
  \draw (0,0.25) to[out=-45+45,in=135-45]
  (0.25,0) to[out=-135+45,in=45-45]
  (0,-0.25) to[out=135+45,in=-45-45]
  (-0.25,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,0.5) to[out=-45+45,in=135-45]
  (0.5,0) to[out=-135+45,in=45-45]
  (0,-0.5) to[out=135+45,in=-45-45]
  (-0.5,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,0.75) to[out=-45+45,in=135-45]
  (0.75,0) to[out=-135+45,in=45-45]
  (0,-0.75) to[out=135+45,in=-45-45]
  (-0.75,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,1) to[out=-45+45,in=135-45]
  (1,0) to[out=-135+45,in=45-45]
  (0,-1) to[out=135+45,in=-45-45]
  (-1,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,1.25) to[out=-45+45,in=135-45]
  (1.25,0) to[out=-135+45,in=45-45]
  (0,-1.25) to[out=135+45,in=-45-45]
  (-1.25,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,1.5) to[out=-45+45,in=135-45]
  (1.5,0) to[out=-135+45,in=45-45]
  (0,-1.5) to[out=135+45,in=-45-45]
  (-1.5,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,1.75) to[out=-45+45,in=135-45]
  (1.75,0) to[out=-135+45,in=45-45]
  (0,-1.75) to[out=135+45,in=-45-45]
  (-1.75,0) to[out=45+45,in=-135-45]
  cycle;
}}]
\path (0,0) pic{axis} (2,2) pic[red,thick,looseness=1]{norm penalty=0};
\node at (2, 2) {\textbullet};
  \draw[blue, thick] (0,1) to[out=-45+45,in=135-45]
  (1,0) to[out=-135+45,in=45-45]
  (0,-1) to[out=135+45,in=-45-45]
  (-1,0) to[out=45+45,in=-135-45]
  cycle;
\end{tikzpicture}
\end{figure}
---

For the lasso problem, we have the constraint $\norm{\beta}_{1} \leq k$. So,
\begin{figure}
\begin{tikzpicture}[pics/axis/.style={code={
  \draw[-] (-2.5,0) -- (2.5,0) node[anchor=north west] {$\beta_1$};
  \draw[-] (0,-2.5) -- (0,2.5) node[anchor=south east] {$\beta_2$};
  }},pics/norm penalty/.style={code={
  \draw (0,0.25) to[out=-45+45,in=135-45]
  (0.25,0) to[out=-135+45,in=45-45]
  (0,-0.25) to[out=135+45,in=-45-45]
  (-0.25,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,0.5) to[out=-45+45,in=135-45]
  (0.5,0) to[out=-135+45,in=45-45]
  (0,-0.5) to[out=135+45,in=-45-45]
  (-0.5,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,0.75) to[out=-45+45,in=135-45]
  (0.75,0) to[out=-135+45,in=45-45]
  (0,-0.75) to[out=135+45,in=-45-45]
  (-0.75,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,1) to[out=-45+45,in=135-45]
  (1,0) to[out=-135+45,in=45-45]
  (0,-1) to[out=135+45,in=-45-45]
  (-1,0) to[out=45+45,in=-135-45]
  cycle;
  \draw (0,1.25) to[out=-45+45,in=135-45]
  (1.25,0) to[out=-135+45,in=45-45]
  (0,-1.25) to[out=135+45,in=-45-45]
  (-1.25,0) to[out=45+45,in=-135-45]
  cycle;
}}]
\path (0,0) pic{axis} (1,2) pic[red,thick,looseness=1]{norm penalty=0};
\node at (1, 2) {\textbullet};
  \draw[blue] (0,1) to[out=-45,in=135]
  (1,0) to[out=-135,in=45]
  (0,-1) to[out=135,in=-45]
  (-1,0) to[out=45,in=-135]
  cycle;
\end{tikzpicture}
\end{figure}
