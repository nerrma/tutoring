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
- Non Parametric Methods
author:
- COMP9417, 22T2
---
# Non Parametric Methods

### Parametric modelling
We make assumptions on the type of function which our data takes.

- Linear regression
- Perceptron
- Logistic regression

### Non parametric modelling

We make no assumptions on the underlying function and purely use our datapoints as guides for pattern inference.

- $k$-Nearest neighbours
- Local regression
- Decision Trees

# Decision Trees

## Entropy

Entropy essentially measures the *uncertainty* or *surprise* of a random variable.

We define the entropy for a set $S$,
\begin{align*}
  H(S) = \sum_{x \in X} -p(x) \log p(x)
\end{align*}
where $p(x)$ represents the *proportion* of $x$ in $S$.

\pause
Say we have a random variable $X \sim$ Bernoulli($p$). We can define the entropy of $X$:

---

\centering
\includegraphics[scale=0.175]{tut4_entropy.png}

## Gain

## Basic Example

## ID3 Algorithm

# $k$-NN


# Extension: Linear Smoothing
