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
- Kernel Methods
author:
- COMP9417, 22T2
---

# Kernel Methods


# Primal vs. Dual Algorithms

The *dual* view of a problem is simply just another way to view a problem mathematically.

\pause

Instead of pure parameter based learning (i.e minimising a loss function etc.), dual algorithms introduce **instance-based** learning.

\pause

This is where we 'remember' mistakes in our data and adjust the corresponding weights accordingly.

We then use a *similarity function* or **kernel** in our predictions to weight the influence of the training data on the prediction.

---

:::: columns
::: column
In the primal problem, we typically learn parameters:

\begin{align*}
  \mathbf{w} \in \mathbb{R}^{p}
\end{align*}

meaning we learn parameters for each of the $p$ features in our dataset.
:::
\pause
::: column
In the dual problem, we typically learn parameters:
\begin{align*}
  \alpha_{i} && \text{ for } i \in [1, n]
\end{align*}
meaning we learn parameters for each of the $n$ **data-points**.

\pause

\vspace{0.5cm}

$\alpha_{i}$ represents the *importance* of a data point $(x_{i}, y_{i})$.
:::
::::

---

**What do we mean by importance?**

\pause

\centering
\includegraphics[scale=0.175]{tut3_linearly_separable.png}

## The Dual/Kernel Perceptron

Recall the *primal* perceptron:

:::: columns
::: column
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
\pause
::: column
If we define the number of iterations the perceptron makes as $K \in \mathbb{N}^{+}$ and assume $\eta = 1$. We can derive an expression for the final weight vector $w^{(K)}$:

\pause

\begin{align*}
  w^{(K)} = \sum_{i=1}^{N} \sum_{j=1}^{K} \mathbf{1}\{y_{i}w^{(j)}x_{i} \leq 0\} y_{i} x_{i}
\end{align*}
:::
::::

---

We can simply our expression and take out the indicator variable:

\begin{align*}
  w^{(K)} &= \sum_{i=1}^{N} \sum_{j=1}^{K} \mathbf{1}\{y_{i}w^{(j)}x_{i} \leq 0\} y_{i} x_{i} \\
   &= \sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}
\end{align*}

where $\alpha_{i}$ is the number of times the perceptron makes a mistake on a data point $(x_{i}, y_{i})$.

---

If we sub in $w^{(K)} = \sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}$. We get the algorithm for the **dual** perceptron.

\vspace{0.5cm}
\pause

\begin{algorithmic}
  \State $converged \gets 0$
  \While{not $converged$}
  \State $converged \gets 1$
  \For{$x_{i} \in X, y_{i} \in y$}
  \If{$y_{i} \sum_{j=1}^{N} \alpha_{j} y_{j} x_{j} \cdot x_{i} \leq 0$}
  \State $\alpha_{i} \gets \alpha_{i} + 1$
  \State $converged \gets 0$
  \EndIf
  \EndFor
  \EndWhile
\end{algorithmic}

---

### Gram Matrix

The Gram matrix represents the *inner product* of two vectors.

For a dataset $X$ we define $G = X^{T} X$. That is:

\pause

\begin{align*}
  G &= \begin{bmatrix}
        \langle x_{1}, x_{1} \rangle & \langle x_{1}, x_{2}\rangle &  \cdots & \langle x_{1}, x_{n} \rangle \\
        \langle x_{2}, x_{1} \rangle & \langle x_{2}, x_{2}\rangle & \cdots & \langle x_{2}, x_{n} \rangle \\
        \vdots & \vdots & \ddots & \vdots  \\
        \langle x_{n}, x_{1} \rangle & \langle x_{n}, x_{2}\rangle &  \cdots & \langle x_{n}, x_{n} \rangle \\
      \end{bmatrix} \\
  G_{i,j} &= \langle x_{i}, x_{j} \rangle
\end{align*}

# Transformations

How do we go about solving **non-linearly separable** datasets with linear classifiers?

\pause

Project them to higher dimensional spaces through a transformation $\phi : \mathbb{R}^{p} \rightarrow \mathbb{R}^{k}$.

\pause

\vspace{-0.5cm}
\centering
\includegraphics[scale=0.33]{tut5_projection.png}

---

Let's revisit the XOR.

\centering
\includegraphics[scale=0.6]{tut5_xor.png}

---

A solution:

\pause

For our input vectors in the form $\textbf{x} = [x_{1}, x_{2}]^{T}$, use a transformation:
\begin{align*}
  \phi(\mathbf{x}) = \begin{bmatrix} 1 \\ \sqrt{2} x_{1} \\ \sqrt{2} x_{2} \\ x_{1}^{2} \\ x_{2}^{2} \\ \sqrt{2} x_{1}x_{2} \end{bmatrix}
\end{align*}

---

For our dataset,

\pause
\begin{align*}
  \phi\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right) = \begin{bmatrix} 1 \\ \sqrt{2} \\ \sqrt{2} \\ 1 \\ 1 \\ \sqrt{2} \end{bmatrix} &
  \phi\left(\begin{bmatrix} -1 \\ -1 \end{bmatrix}\right) = \begin{bmatrix} 1 \\ -\sqrt{2} \\ -\sqrt{2} \\ 1 \\ 1 \\ \sqrt{2} \end{bmatrix} &&
  \phi\left(\begin{bmatrix} -1 \\ 1\end{bmatrix}\right) = \begin{bmatrix} 1 \\ -\sqrt{2} \\ \sqrt{2} \\ 1 \\ 1 \\ -\sqrt{2} \end{bmatrix} &
  \phi\left(\begin{bmatrix} 1 \\ -1\end{bmatrix}\right) = \begin{bmatrix} 1 \\ \sqrt{2} \\ -\sqrt{2} \\ 1 \\ 1 \\ -\sqrt{2} \end{bmatrix} &
\end{align*}

---

:::: columns
::: column
For the negative class:
\begin{align*}
  \phi\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right)_{2,6} &= \begin{bmatrix} \sqrt{2} \\ \sqrt{2} \end{bmatrix} \\
  \phi\left(\begin{bmatrix} -1 \\ -1 \end{bmatrix}\right)_{2,6} &= \begin{bmatrix} -\sqrt{2} \\ \sqrt{2} \end{bmatrix} &
\end{align*}
:::
::: column
For the positive class:
\begin{align*}
  \phi\left(\begin{bmatrix} -1 \\ 1 \end{bmatrix}\right)_{2,6} &= \begin{bmatrix} -\sqrt{2} \\ -\sqrt{2} \end{bmatrix} \\
  \phi\left(\begin{bmatrix} 1 \\ -1 \end{bmatrix}\right)_{2,6} &= \begin{bmatrix} \sqrt{2} \\ -\sqrt{2} \end{bmatrix}
\end{align*}
:::
::::

\pause
\centering
\includegraphics[scale=0.35]{tut5_xor_projected.png}

---

We may have a problem, recall the **dual perceptron**.

\begin{algorithmic}
  \State $converged \gets 0$
  \While{not $converged$}
  \State $converged \gets 1$
  \For{$x_{i} \in X, y_{i} \in y$}
  \only<1>{\If{$y_{i} \sum_{j=1}^{N} \alpha_{j} y_{j} x_{j} \cdot x_{i} \leq 0$}}
  \only<2>{\If{$y_{i} \sum_{j=1}^{N} \alpha_{j} y_{j} \phi(x_{j}) \cdot \phi(x_{i}) \leq 0$}}
  \State $\alpha_{i} \gets \alpha_{i} + 1$
  \State $converged \gets 0$
  \EndIf
  \EndFor
  \EndWhile
\end{algorithmic}

---

Recall the transformation $\phi : \mathbb{R}^{p} \rightarrow \mathbb{R}^{k}$. \pause For an arbitrarily large $k$,
\begin{align*}
  G &= \begin{bmatrix}
        \langle \phi(x_{1}), \phi(x_{1}) \rangle & \langle \phi(x_{1}), \phi(x_{2})\rangle &  \cdots & \langle \phi(x_{1}), x_{n} \rangle \\
        \langle \phi(x_{2}), \phi(x_{1}) \rangle & \langle \phi(x_{2}), \phi(x_{2})\rangle & \cdots & \langle \phi(x_{2}), \phi(x_{n}) \rangle \\
        \vdots & \vdots & \ddots & \vdots  \\
        \langle \phi(x_{n}), \phi(x_{1}) \rangle & \langle \phi(x_{n}), \phi(x_{2})\rangle & \cdots & \langle \phi(x_{n}), \phi(x_{n}) \rangle \\
      \end{bmatrix} \\
\end{align*}

the Gram matrix becomes far too complex to compute.

# The Kernel Trick

An absolute mathemagical idea which allows us to calculate the values of the Gram matrix for cheap.

Recall the transformation to the XOR data:
\begin{align*}
  \only<1>{\phi(\mathbf{x}) = \begin{bmatrix} 1 \\ \sqrt{2} x_{1} \\ \sqrt{2} x_{2} \\ x_{1}^{2} \\ x_{2}^{2} \\ \sqrt{2} x_{1}x_{2} \end{bmatrix}}
  \only<2>{\phi(\mathbf{x}) \cdot \phi(\mathbf{y}) = \begin{bmatrix} 1 \\ \sqrt{2} x_{1} \\ \sqrt{2} x_{2} \\ x_{1}^{2} \\ x_{2}^{2} \\ \sqrt{2} x_{1}x_{2} \end{bmatrix} \begin{bmatrix} 1 \\ \sqrt{2} y_{1} \\ \sqrt{2} y_{2} \\ y_{1}^{2} \\ y_{2}^{2} \\ \sqrt{2} y_{1}y_{2} \end{bmatrix}}
\end{align*}

---

\begin{align*}
  \only<1->{\phi(\mathbf{x}) \cdot \phi(\mathbf{y}) &= 1 + 2x_{1} y_{1} + 2 x_{2} y_{2} + x_{1}^{2} y_{1}^{2} + x_{2}^{2} y_{2}^{2} + 2 x_{1} x_{2} y_{1} y_{2}} \\
  \only<2->{\phi(\mathbf{x}) \cdot \phi(\mathbf{y}) &= 1 + 2(x_{1} y_{1} + x_{2} y_{2}) + (x_{1} y_{1} + x_{2} y_{2})^{2} }\\
  \only<3->{\phi(\mathbf{x}) \cdot \phi(\mathbf{y}) &= (1 + \mathbf{x} \cdot \mathbf{y})^{2}}
\end{align*}

\pause
\pause
\pause

Say we define a *kernel*: $k(\mathbf{x}, \mathbf{y}) = (1 + \mathbf{x} \cdot \mathbf{y})^{2}$

\pause

So our Gram matrix is:

\vspace{-0.5cm}
\begin{align*}
  G &= \begin{bmatrix}
        k(x_{1}, x_{1}) &  k(x_{1}, x_{2})&  \cdots &  k(x_{1}, x_{n}) \\
        k(x_{2}, x_{1}) &  k(x_{2}, x_{2})&  \cdots &  k(x_{2}, x_{n}) \\
        \vdots & \vdots & \ddots & \vdots  \\
        k(x_{n}, x_{1}) &  k(x_{n}, x_{2}) &  \cdots &  k(x_{n}, x_{n}) \\
      \end{bmatrix} \\
\end{align*}

\pause
\vspace{-0.7cm}
**Why is this useful?**

# Support Vector Machines
