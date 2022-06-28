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

## Decision Trees

A tree-like model used for both **regression** and **classification**.

\pause

Advantages:

\pause

- Interpretable
- Useful when used in ensemble learning (we'll come back to this notion)

\pause

Disadvantages:

\pause

- Tend to overfit data
- Often innacurate in their most basic form

## Entropy

Entropy essentially measures the *uncertainty* or *surprise* of a random variable.

We define the entropy for a set $S$,
\begin{align*}
  H(S) = \sum_{x \in X} -p(x) \log p(x)
\end{align*}
where $p(x)$ represents the *proportion* of $x$ in $S$.

\pause
Say we have a random variable $X \sim$ Bernoulli($p$). We can define the entropy of $X$:

\pause
\begin{align*}
  H(x) = -(1-p) \log (1-p) - p \log p
\end{align*}

---

\centering
\includegraphics[scale=0.175]{tut4_entropy.png}

## Gain

To measure the *information* we gain by splitting on an attribute $A$ for a dataset $S$, we define:

\pause
\begin{align*}
  \text{Gain}(S, A) = \text{Current entropy} - \text{Entropy if we split on A} \\
\end{align*}

\pause
If we have a dataset $S$ with a feature $A$,
\begin{align*}
  \text{Gain}(S, A) = H(S) - \sum_{v \in V_{A}} \frac{|S_{v}|}{|S|} H(S_{v}) \\
\end{align*}


## Basic Example

Say we have a dataset as follows: $[29+, 35-]$:

- A1 ~ T: $[21+, 5-]$ F: $[8+, 30-]$
- A2 ~ T: $[18+, 33-]$ F: $[11+, 2-]$

\pause
\begin{align*}
  H(S) &= \sum_{x \in X} -p(x) \log p(x) \\
  &= -\frac{29}{29+35} \log (\frac{29}{29+35}) - \frac{35}{29+35} \log(\frac{35}{29+35}) \\
  &= 0.9936
\end{align*}

---

:::: columns
::: column
Dataset: $[29+, 35-]$:

- A1 ~ T: $[21+, 5-]$ F: $[8+, 30-]$
:::
::: column
\begin{align*}
  H(S) = 0.9936
\end{align*}
:::
::::

\pause

\begin{align*}
  H(S_{A_{1,T}}) &= -\frac{21}{26} \log(\frac{21}{26}) -\frac{5}{26} \log(\frac{5}{26}) \\
   &= 0.7063
\end{align*}

\pause

\begin{align*}
  H(S_{A_{1,F}}) &= -\frac{8}{38} \log(\frac{8}{38}) -\frac{30}{38} \log(\frac{30}{38}) \\
   &= 0.7425
\end{align*}

---

:::: columns
::: column
Dataset: $[29+, 35-]$:

- A2 ~ T: $[18+, 33-]$ F: $[11+, 2-]$
:::
::: column
\begin{align*}
  H(S) = 0.9936 \\
  H(S_{A_{1,T}}) = 0.7063 \\
  H(S_{A_{1,F}}) = 0.7425 \\
\end{align*}
:::
::::

\pause

\begin{align*}
  H(S_{A_{2,T}}) &= -\frac{18}{51} \log(\frac{18}{51}) -\frac{33}{51} \log(\frac{33}{51}) \\
   &= 0.9366
\end{align*}

\pause

\begin{align*}
  H(S_{A_{2,F}}) &= -\frac{11}{13} \log(\frac{11}{13}) -\frac{2}{13} \log(\frac{2}{13}) \\
   &= 0.4674
\end{align*}

---

:::: columns
::: column
Dataset: $[29+, 35-]$:

- A1 ~ T: $[21+, 5-]$ F: $[8+, 30-]$
- A2 ~ T: $[18+, 33-]$ F: $[11+, 2-]$
:::
::: column
\begin{align*}
  H(S) = 0.9936 \\
  H(S_{A_{1,T}}) = 0.7063 \\
  H(S_{A_{1,F}}) = 0.7425 \\
  H(S_{A_{2,T}}) = 0.9366 \\
  H(S_{A_{2,F}}) = 0.4674 \\
\end{align*}
:::
::::

\pause

\begin{align*}
 \text{Gain}(S, A_1) &= H(S) - \sum_{v \in \{T, F\}}  \frac{|A_{1,v}|}{|S|} H(A_{1,v}) \\
\end{align*}


\vspace{-1.5cm}
\pause

\begin{align*}
 &= H(S) - \frac{26}{64} H(A_{1,T}) -  \frac{38}{64} H(A_{1,F}) \\
 &= 0.2658
\end{align*}

---

:::: columns
::: column
Dataset: $[29+, 35-]$:

- A1 ~ T: $[21+, 5-]$ F: $[8+, 30-]$
- A2 ~ T: $[18+, 33-]$ F: $[11+, 2-]$

\begin{align*}
 \text{Gain}(S, A_1) = 0.2658
\end{align*}

:::
::: column
\begin{align*}
  H(S) = 0.9936 \\
  H(S_{A_{1,T}}) = 0.7063 \\
  H(S_{A_{1,F}}) = 0.7425 \\
  H(S_{A_{2,T}}) = 0.9366 \\
  H(S_{A_{2,F}}) = 0.4674 \\
\end{align*}
:::
::::

\pause

\begin{align*}
 \text{Gain}(S, A_2) &= H(S) - \frac{51}{64} H(A_{2,T}) -  \frac{13}{64} H(A_{2,F}) \\
 &= 0.1643
\end{align*}

## ID3 Algorithm

Basically what we just did:

- Calculate the entropy for each attribute $a \in A$.
- Split on the attribute with the maximum Gain. This means creating a decision tree node using that attribute.
- Recurse on this new subset of the data.

## Regression Trees

Regression trees split the dataset up into regions and fit separate models to each region.

:::: columns
::: column
\pause
\centering
\includegraphics[scale=0.4]{tut4_treg_tree.png}
:::
::: column
\pause
\centering
\includegraphics[scale=0.4]{tut4_treg_regions.png}
:::
::::

---

If we define our two regions as $R_{1}(j, s) = \{X|X_{j} \leq s\}$ and $R_{2}(j, s) = \{X|X_{j} > s\}$. We can find optimal regions with the formula:
\pause

\begin{align*}
  \min_{j,s} \left[\min_{c_{1}} \sum_{x_{i} \in R_{1}(j, s)} (y_{i} - c_{1})^{2} + \sum_{x_{i} \in R_{2}(j, s)}\min_{c_{2}} (y_{1} - c_{2})^{2} \right]
\end{align*}

Where $\hat{c_{1}} = \text{ave}(y_{i} | x_{i} \in R_{1}), \quad \hat{c_{2}} = \text{ave}(y_{i} | x_{i} \in R_{2})$.

\pause
This essentially finds regions ($R_{1}$ and $R_{2}$) with the minimum variance.

---

\centering
\includegraphics[scale=0.6]{tut4_treg_data.png}
\begin{align*}
  \min_{j,s} \left[\min_{c_{1}} \sum_{x_{i} \in R_{1}(j, s)} (y_{i} - c_{1})^{2} + \sum_{x_{i} \in R_{2}(j, s)}\min_{c_{2}} (y_{1} - c_{2})^{2} \right]
\end{align*}


# $k$-NN

We predict $\hat{y}_{i}$ for a point $x_{i}$ to be the average of the $k$-nearest points.
\vspace{1.5cm}

:::: columns
::: column

\pause
**Regression** If we define the set $K$ as the $k$-nearest neighbours of a point $X_{i}$, then our $k$-NN estimate is:
\begin{align*}
  \hat{y}_{i} = \frac{1}{k} \sum_{i=1}^{n} \mathbf{1}\{X_{i} \in K\} y_{i}
\end{align*}

:::
::: column

\pause
**Classification** we assign $X_{i}$ the majority class in $K$.
:::
::::

---

\centering
\includegraphics[scale=0.2]{tut4_knn.png}

---

**An obvious limitation:**
\pause

\begin{center}
We need lots of relevant data for accurate predictions.
\end{center}

\pause

**A not-so obvious limitation:**
\pause

\begin{center}
Curse of dimensionality.
\end{center}

---

**Curse of Dimensionality**

Certain phenomena occur when we increase the number of dimensions (i.e features) in our problem.

The most common are:

\pause
- Distances between points breaking down
\pause
- The need for even *more* data


---

\centering
\includegraphics[scale=0.275]{tut4_dimhist.png}

---

Say we have a classification task and we want 10 samples per unique combination of variables for a comprehensive data set.

\pause

- *1 binary variable* \pause - 2 unique combinations \pause - 20 samples \pause
- *2 binary variables* - 4 unique combinations - 40 samples \pause
- *k binary variables* - $2^k$ unique combinations - $10 \times 2^{k}$ samples

\pause

\vspace{2cm}

So for $20$ features, we need $10 \times 2^{20} = 10485760$ data points!


# Linear Smoothing


$k$-NN regression typically fits a choppy model to our data. Linear smoothing tries to smooth out the fit by incorporating a *kernel* to weight the influence nearest neighbours by distance.

\pause

If we define $h$ as the smoothing parameter and $K$ as the kernel, the Linear Smoothing estimate is:
\begin{align*}
  \hat{y}_{i} =  \frac{\sum_{j=1}^{n} K\left(\frac{\norm{x_{i}-x_{j}}}{h}\right) y_{i}}{\sum_{j=1}^{n} K\left(\frac{\norm{x_{i}-x_{j}}}{h}\right)} \\
\end{align*}

\pause

\vspace{-0.5cm}

As $h \to 0$ our distances have a higher variance. If $h \to \infty$ have a lower variance, and our model is in turn smoother.

---
:::: columns
::: column

\centering
\includegraphics[scale=0.310]{tut4_kernel_nn.png}

\pause

:::
::: column

\centering
\includegraphics[scale=0.315]{tut4_kernel_para.png}
:::
::::
