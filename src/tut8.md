---
header-includes: |
	\usepackage{amsmath}
	\usepackage{fancyhdr}
	\usepackage{physics}
	\usepackage{hyperref}
	\usepackage{pgfplots}
	\usepackage{algpseudocode}
	\usepackage{graphicx}
	\usepackage{tikz}
		\usetikzlibrary{positioning}
	\graphicspath{ {./images/} }
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
	\tikzset{basic/.style={draw,fill=blue!20,text width=1em,text badly centered}}
	\tikzset{input/.style={basic,circle}}
	\tikzset{weights/.style={basic,rectangle}}
	\tikzset{functions/.style={basic,circle,fill=blue!10}}
title:
- Unsupervised Learning + Revision
author:
- COMP9417, 22T2
---

# Unsupervised Learning

Learning without any labels.

For example,
-- Cluster analysis (i.e grouping users of a social media, classifying similar events/data without knowing any other information)
-- Signal separation (i.e PCA, SVD)

\pause
The content this week is light, so I'll go straight to the lab to explain it.

# Revision

## Identities

Of course you need to remember anything from first year/high school mathematics (i.e basis calculus, log laws, basic vector/matrix identities).

\pause
Some general identities which may be useful for this course:

### Vector Calculus

If $x$ is an arbitrary vector, and $c$ is any constant (vector or scalar),
\begin{align*}
  \frac{\partial(xc)}{\partial x} &= c^{T} && \frac{\partial(x^{T}cx)}{\partial x} = 2cx
\end{align*}

## The First Question

What is this problem, and how do we solve it?

\begin{align*}
  \hat{\beta} = \argmin_{\beta} \norm{y - X\beta}_{2}^{2}
\end{align*}

\pause
Describe Ridge and LASSO regression and how they differ.

## Linear Methods

Name this algorithm and what it represents,

\begin{align*}
  \hat{p} &= \sigma(X\beta) \\
  &= \frac{1}{1 + e^{-X\beta}} \\
\end{align*}

## Dual Perceptron

:::: columns
::: column
Recall the primal perceptron:
\begin{algorithmic}
  \State $converged \gets 0$
  \While{not $converged$}
  \State $converged \gets 1$
  \For{$x_{i} \in X, y_{i} \in y$}
  \If{$y_{i} w \cdot x_{i} \leq 0$}
  \State $w \gets w + \eta y_{i} x_{i}$
  \State $converged \gets 0$
  \EndIf
  \EndFor
  \EndWhile
\end{algorithmic}
:::
::: column
\pause
- How did we derive the dual perceptron?

\pause
- What is the **Kernel trick**?

\pause
- What problem does the SVM solve?
:::
::::

## Ensemble Methods

Describe the difference between bagging and boosting.

\pause

Why does bagging reduce our model's variance?

## Neural Learning

Given the following diagram, derive expressions for $\frac{\partial L}{\partial \theta_{k}}$ for $k = 0, \ldots, K$ where $\theta_{k} = \{ A_{k}, b_{k}\}$

\centering{\includegraphics[scale=0.5]{tut7_nn_arch.png}}

# Gradient Descent Question

Given $w=(w_{0}, w_{1}, w_{2}, w_{3})^{T}$, $X^{(i)} = (1, x_{1}^{(i)}, x_{2}^{(i)}, x_{3}^{(i)})$ for a model:

\begin{align*}
  \hat{y}^{i} &= w_{0} + w_{1} x_{1}^{(i)} + w_{2} x_{2}^{(i)} + w_{3} x_{3}^{(i)}
  \hat{y}^{i} &= w^{T} X^{(i)}
\end{align*}

We define the mean-loss of our model as:
\begin{align*}
  L_{c}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} L_{c}(y^{(i)}, \hat{y}^{(i)}) = \frac{1}{n} \sum_{i=1}^{n} \left[ \sqrt{\frac{1}{c^{2}} (y^{(i)} - \langle w^{(t)}, X^{(i)}\rangle)^{2} + 1} - 1 \right]
\end{align*}

## Part A

Calculate $\frac{\partial L_{c}(y, \hat{y})}{\partial w_{k}}$, where $k = 0, \ldots, 4$.

## Part B

Take $c=2$, what are the GD updates to $w$ for a learning rate $\eta$? What are the SGD updates?

\pause
\begin{align*}
  w_{k}^{(t+1)} =   w_{k}^{(t)} - \eta\cdot\frac{1}{n} \sum_{i=1}^{n} \frac{X_{k}^{(i)}(y_{i} - \langle w^{(t)}, X^{(i)}\rangle)}{2\sqrt{(y_{i} - \langle w^{(t)}, X^{(i)}\rangle)^{2} + 4}}
\end{align*}

\pause
For SGD,
\begin{align*}
  w_{k}^{(t+1)} =   w_{k}^{(t)} - \frac{X_{k}^{(i)}(y_{i} - \langle w^{(t)}, X^{(i)}\rangle)}{2\sqrt{(y_{i} - \langle w^{(t)}, X^{(i)}\rangle)^{2} + 4}} && \text{for a random } i \in [1, n]
\end{align*}
