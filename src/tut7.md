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
- Neural Learning
author:
- COMP9417, 22T2
---

# Neural Learning

You'll typically see this field referred to as *deep learning*.

Deep learning has become the forefront of modern machine learning. With it comes many challenges and intricacies which are out of the scope of this course (see COMP9444, *Deep Learning Book* by Goodfellow et al).

\pause

This course discuses what makes up neural networks, partially why they are effective and how they work.

# Recap: The Perceptron

\centering
\begin{tikzpicture}
	\node[functions] (center) {};
	\node[below of=center,font=\scriptsize,text width=4em] {Activation function};
	\draw[thick] (0.5em,0.5em) -- (0,0.5em) -- (0,-0.5em) -- (-0.5em,-0.5em);
	\draw (0em,0.75em) -- (0em,-0.75em);
	\draw (0.75em,0em) -- (-0.75em,0em);
	\node[right of=center] (right) {};
		\path[draw,->] (center) -- (right);
	\node[functions,left=3em of center] (left) {$\sum$};
    \path[draw,->] (left) -- (center);
	\node[weights,left=3em of left] (2) {$w_2$} -- (2) node[input,left of=2] (l2) {$x_2$};
		\path[draw,->] (l2) -- (2);
		\path[draw,->] (2) -- (left);
	\node[below of=2] (dots) {$\vdots$} -- (dots) node[left of=dots] (ldots) {$\vdots$};
	\node[weights,below of=dots] (n) {$w_n$} -- (n) node[input,left of=n] (ln) {$x_n$};
		\path[draw,->] (ln) -- (n);
		\path[draw,->] (n) -- (left);
	\node[weights,above of=2] (1) {$w_1$} -- (1) node[input,left of=1] (l1) {$x_1$};
		\path[draw,->] (l1) -- (1);
		\path[draw,->] (1) -- (left);
	\node[weights,above of=1] (0) {$w_0$} -- (0) node[input,left of=0] (l0) {$1$};
		\path[draw,->] (l0) -- (0);
		\path[draw,->] (0) -- (left);
	\node[below of=ln,font=\scriptsize] {inputs};
	\node[below of=n,font=\scriptsize] {weights};
\end{tikzpicture}

# Multi-layer Perceptron

A multi-layer perceptron is where we *chain* these perceptrons to learn non-linear patterns.

:::: columns
::: column
\pause
\centering
\includegraphics[scale=0.45]{tut7_nn_single.png}
:::
::: column
\pause

If we define the activation function used for the hidden layer as $g$ and the weights for input features as $\beta$:
\begin{align*}
  f(X) &= w_{0} + \sum_{i=1}^{n} w_{i} A_{i} \\
  \only<3>{&= w_{0} + \sum_{i=1}^{n} w_{i} g(X_{i}) \\}
  \only<4->{&= w_{0} + \sum_{i=1}^{n} w_{i} g(\beta_{0} + \sum_{j=1}^{p} \beta_{j} X_{i}) \\}
\end{align*}
:::
::::

---

\centering
\includegraphics[scale=0.45]{tut7_nn_multi.png}

# Back-propagation

The main problem now becomes: **How do we learn this large number of weights?**

\pause

As always, we define an appropriate loss function and optimise it. \pause Due to the complexity of the *function* which is the neural network, we'll need to perform gradient descent to gradually improve our model over time.

\pause

But how do we even calculate the gradient?

---

:::: columns
::: column
\begin{tikzpicture}
	\node[functions] (center) {};
	\node[below of=center,font=\scriptsize] {$a$};
	\draw[thick] (0.5em,0.5em) -- (0,0.5em) -- (0,-0.5em) -- (-0.5em,-0.5em);
	\draw (0em,0.75em) -- (0em,-0.75em);
	\draw (0.75em,0em) -- (-0.75em,0em);
	\node[functions, right=2.5em] {$L$};
	\node[right of=center] (right) {};
		\path[draw,->] (center) -- (right);
	\node[functions,left=3em of center] (left) {$\sum$};
    \path[draw,->] (left) -- (center);
	\node[weights,left=3em of left] (2) {$w_2$} -- (2) node[input,left of=2] (l2) {$x_2$};
		\path[draw,->] (l2) -- (2);
		\path[draw,->] (2) -- (left);
	\node[below of=2] (dots) {$\vdots$} -- (dots) node[left of=dots] (ldots) {$\vdots$};
	\node[weights,below of=dots] (n) {$w_n$} -- (n) node[input,left of=n] (ln) {$x_n$};
		\path[draw,->] (ln) -- (n);
		\path[draw,->] (n) -- (left);
	\node[weights,above of=2] (1) {$w_1$} -- (1) node[input,left of=1] (l1) {$x_1$};
		\path[draw,->] (l1) -- (1);
		\path[draw,->] (1) -- (left);
	\node[weights,above of=1] (0) {$w_{0}$} -- (0) node[input,left of=0] (l0) {$1$};
		\path[draw,->] (l0) -- (0);
		\path[draw,->] (0) -- (left);
	\node[below of=ln,font=\scriptsize] {inputs};
	\node[below of=n,font=\scriptsize] {weights};
\end{tikzpicture}
:::
::: column
\pause
The loss is a function of the activation:
\begin{align*}
  L(a, y)
\end{align*}

\pause

The activation is a function of the inputs, the weights and the bias:
\begin{align*}
  a(x_{1}, \ldots, x_{n}, w_{0}, \ldots, w_{n})
\end{align*}

\pause

What we want to optimise the loss function is:

\begin{align*}
  \only<4>{\frac{\partial L}{\partial w_{i}}}
  \only<5->{\frac{\partial L}{\partial w_{i}} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial w_{i}}}
\end{align*}

:::
::::

---

Say we have the following network architecture, where $A$ represents the weights and $b$ the bias:

\centering{\includegraphics[scale=0.5]{tut7_nn_arch.png}}

If we say that $\theta_{K} = \{ A_{K}, b_{K} \}$ for a layer $k$. The gradient of our coefficients looks like this:
\begin{align*}
  \only<1>{\frac{\partial L}{\partial \theta_{K-1}} = \frac{\partial L}{\partial f_{K}} \frac{\partial f_{K}}{\partial \theta_{K-1}}}
  \only<2>{\frac{\partial L}{\partial \theta_{K-2}} = \frac{\partial L}{\partial f_{K}} \frac{\partial f_{K}}{\partial f_{K-1}} \frac{\partial f_{K-1}}{\partial \theta_{K-2}}}
  \only<3>{\frac{\partial L}{\partial \theta_{K-3}} = \frac{\partial L}{\partial f_{K}} \frac{\partial f_{K}}{\partial f_{K-1}} \frac{\partial f_{K-1}}{\partial f_{K-2}} \frac{\partial f_{K-2}}{\partial \theta_{K-3}}}
\end{align*}

---

Using back-propagation, we can therefore calculate:

\begin{align*}
  \nabla L(\mathbf{\theta}, y) = \left[ \frac{\partial L}{\partial \theta_{1}}, \frac{\partial L}{\partial \theta_{2}}, \ldots, \frac{\partial L}{\partial \theta_{K}}\right] \\
\end{align*}
\pause

We can then all of our parameters (in the basic case):
\begin{align*}
  \mathbf{\theta}^{(t)} = \mathbf{\theta}^{(t-1)} - \nabla L(\theta, y) \\
\end{align*}

Typically, the optimiser will be some form of stochastic gradient descent (minibatch in some cases) as classic gradient descent is expensive for a large number of parameters and data points.

----

Let's visualise a neural network working:

\centering

\href{https://playground.tensorflow.org/}{Tensorflow Playground}

# Reminder: Revision

\centering
Next week is the last week of tutorials!

Are there any specific topics you want resources or revision on?
