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
title:
- Regression I
author:
- COMP9417, 22T2
---

# What do you prefer?

More theory, more practice (i.e Python and using packages), going through questions, consultation etc.

# Thinking Machine Learning

We try to make sense of data using mathematics to help us quantify what we *know*.

A standard way to break the problem down is as follows:

- We have 'input' data $X$ and targets/outputs $y$
- Our data can be modelled as $y = f(X)$
- Goal is to find the best approximation for $f$ as $\hat{f}$


We define the quality of our approximation ($\hat{f}$) by using a error/loss function.

# Linear Regression

::: columns

:::: column

We deduct and assume a linear relationship between $X$ and $y$. 

\vspace{4pt}

In this simple case, our model will take the form:

\begin{center}
$\hat{y} = w_0 + w_1 X$
\end{center}

\vspace{8pt}

**How do we find the optimal $w_0$ and $w_1$?**

::::

:::: column

\includegraphics[scale=0.4]{tut1_data.png}

::::

:::

---

What will our loss function need?

~ Boils down to the properties of the target function.

- Target function has $\approx 0$ distance to all points
- We can define a basic loss function with one glaring issue:

\begin{align*}
	L(w_0, w_1) &= \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})
\end{align*}


---

To make life easy, we define our loss function as:

\begin{align*}
	L(w_0, w_1) &= \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})^2 & \text{ a.k.a MSE } \\
	&= \frac{1}{n} \sum_{i=1}^n (y_i - w_0 - w_1 x_i)^2 & \text{ by definition }
\end{align*}

The minimum of our loss function w.r.t $w_0$ and $w_1$ will be their optimal values respectively.

# Question 1 (a, b, c)

## 1a

Derive the least-squares estimates for the univariate linear regression model.

i.e Solve:
\begin{align*}
	\argmin_{w_0, w_1}& L(w_0, w_1) \\
	\argmin_{w_0, w_1}& \frac{1}{n} \sum_{i=1}^n (y_i - w_0 - w_1 x_i)^2 \\
\end{align*}

---

First we differentiate $L(w_0, w_1)$ with respect to $w_0$, 
\begin{align*}
	\frac{\partial L(w_0, w_1)}{\partial w_0} &= -\frac{2}{n} \sum_{i=1}^n (y_i - w_0 - w_1 x_i) \\
	&= -\frac{2}{n} \left( \sum_{i=1}^n y_i - n w_0 - w_1 \sum_{i=1}^n x_i \right) \\
\end{align*}

For the minimum, $\frac{\partial L(w_0, w_1)}{\partial w_0} = 0$,
\begin{align*}
	-\frac{2}{n} \left( \sum_{i=1}^n y_i - n w_0 - w_1 \sum_{i=1}^n x_i \right) &= 0\\
\end{align*}

---

\begin{equation}
\begin{aligned}
	\frac{1}{n} \sum_{i=1}^n y_i - w_0 - w_1 \frac{1}{n} \sum_{i=1}^n x_i &= 0\\
	\bar{y} - w_0 - w_1 \bar{x} &= 0\\
	w_0 = \bar{y} - w_1 \bar{x}\\
\end{aligned}
\end{equation}

To find $w_1$, we follow a similar process and use simple simultaneous equations to solve for the final solution.

---

So,
\begin{align*}
	\frac{\partial L(w_0, w_1)}{\partial w_1} &= -\frac{2}{n} \sum_{i=1}^n x_i (y_i - w_0 - w_1 x_i) \\
	&= -\frac{2}{n} \left( \sum_{i=1}^n x_i y_i - w_0 \sum_{i=1}^n x_i - w_1 \sum_{i=1}^n x_i^2  \right) \\
\end{align*}

$\frac{\partial L(w_0, w_1)}{\partial w_1} = 0$,
\begin{align*}
	\frac{1}{n} \left( \sum_{i=1}^n x_i y_i -  w_0\sum_{i=1}^n x_i - w_1\sum_{i=1}^n x_i^2  \right) = 0 \\
	\overline{xy} -  w_0 \bar{x} - w_1 \overline{x^2} = 0 \\
\end{align*}

---

\begin{equation}
\begin{aligned}
	\overline{xy} -  w_0 \bar{x} - w_1 \overline{x^2} &= 0 \\
	w_1 &= \frac{\overline{xy} -  w_0 \bar{x}}{\overline{x^2}}\\
\end{aligned}
\end{equation}

Sub (1) into (2):
\begin{align*}
	w_1 &= \frac{\overline{xy} -  (\bar{y} - w_1 \bar{x}) \bar{x}}{\overline{x^2}}\\
	w_1 &= \frac{\overline{xy} -  \bar{x}\bar{y} +  w_1 \bar{x}^2}{\overline{x^2}}\\
	w_1 (\frac{\overline{x^2} - \bar{x}^2}{\bar{x}^2}) &= \frac{\overline{xy} -  \bar{x}\bar{y} +  w_1 \bar{x}^2}{\overline{x^2}}\\
	w_1 &= \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2}
\end{align*}

---

Finally, we have

\begin{align*}
	w_1 &= \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} \text{ and } w_0 = \bar{y} - w_1 \bar{x}
\end{align*}


## 1b

**Problem**: Prove $(\bar{x}, \bar{y})$ is on the line.

From 1(a), the equation of our line ($\hat{y} = w_0 + w_1 x$) becomes:

\begin{align*}
	\hat{y} &= \bar{y} - \bar{x} \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} + \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} x \\
\end{align*}

Sub $x = \bar{x}$, 
\begin{align*}
	\hat{y} &= \bar{y} - \bar{x} \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} + \frac{\overline{xy} -  \bar{x}\bar{y}}{\overline{x^2} - \bar{x}^2} \bar{x} \\
	\hat{y} &= \bar{y} & \therefore (\bar{x}, \bar{y}) \text{ is on the line }
\end{align*}

## 1c

Similar to 1a, though take care with the partial derivatives:

\begin{align*}
	\frac{\partial L(w_0, w_1)}{\partial w_0} = -\frac{2}{n} \sum_{i=1}^n (y_i - w_0 - w_1 x_i) \\
	\frac{\partial L(w_0, w_1)}{\partial w_1} = -\frac{2}{n} \sum_{i=1}^n x_i (y_i - w_0 - w_1 x_i) + 2 \lambda w_1
\end{align*}

---

Final result is:

\begin{align*}
	w_0 = \bar{y} - w_1 \bar{x} \\
	w_1 = \frac{\overline{xy} - \bar{x} \bar{y}}{\overline{x^2} - \bar{x}^2 + \lambda}
\end{align*}

Notice how the coefficients have an inverse relationship with $\lambda$.


# Question 2 (a, b, c, d)

## 2a

**Problem**: Show that $\mathcal{L}(w) = \frac{1}{n} \norm{y - Xw}^2_2$ has critical point $\hat{w} = (X^T X)^{-1} X^T y$.

To find optimal $w$, solve $\displaystyle \frac{\partial\mathcal{L}(w)}{\partial w} = 0$

\begin{align*}
	\mathcal{L}(w) &= \frac{1}{n} (y - Xw)^T (y - Xw)\\
	&= \frac{1}{n} \left( y^T y - y^T X w  - w^T X^T y + w^T X^T X w \right) \\
	&= \frac{1}{n} \left( y^T y - 2y^T X w + w^T X^T X w \right) \\
\end{align*}

\begin{align*}
	\frac{\partial\mathcal{L}(w)}{\partial w} &= -\frac{1}{n} (-2 X^t y + 2 X^T X w) \\
\end{align*}

---

To solve for $\hat{w}$,

\begin{align*}
	- 2 X^t y + 2 X^T X \hat{w} = 0\\
	\hat{w} = (X^T X)^{-1} X^T y\\
\end{align*}

## 2b
**Problem**: Prove $\hat{w} = (X^T X)^{-1} X^T y$ is a global minimum.
\begin{align*}
	\nabla_w^2 \mathcal{L}(w) &= \nabla_w (\nabla_w \mathcal{L}(w)) \\
	&= \nabla_w (-2X^T y + 2X^T X w) \\
	&= 2 X^T X
\end{align*}

So, for a vector $u \in \mathbb{R}^p$, 
\begin{align*}
	u^T (2 X^T X) u &= 2(u^TX^T)(Xu) \\
	&= 2(u^TX^T)(Xu) \\
	&= 2(Xu)^T(Xu) \\
	&= 2\norm{Xu}^2_2 \geq 0 \\
\end{align*}

---

Therefore, $\mathcal{L}$ is convex and $\hat{w}$ is the unique global minimum.

## 2c

$x_i = \begin{bmatrix} 1 & x_{i1} \end{bmatrix}$ to represent our input \& the bias ($w_0$)

$y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$ to represent the target variable

$w = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}$ to represent the parameters

---

::: columns

:::: column
\begin{equation*}
\begin{aligned}
	X &= \begin{bmatrix} 1 & x_{11} \\ 1 & x_{21} \\ \vdots & \vdots \\1 & x_{n1} \end{bmatrix} \\
	X^T y &= \begin{bmatrix} 1 & 1 &\cdots& 1 \\ x_{11} &x_{21} &\cdots &x_{n1}  \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}\\
	X^T y &= \begin{bmatrix} n \bar{y} \\ n \overline{xy} \end{bmatrix}
\end{aligned}
\end{equation*}
::::

:::: column
\begin{equation*}
\begin{aligned}
	X^T X &= \begin{bmatrix} 1 & 1 &\cdots& 1 \\ x_{11} &x_{21} &\cdots &x_{n1}  \end{bmatrix} \begin{bmatrix} 1 & x_{11} \\ 1 & x_{11} \\ \vdots & \vdots \\1 & x_{n1} \end{bmatrix} \\
	&= \begin{bmatrix} n & \sum_{i=1}^n x_i \\ \sum_{i=1}^n x_i & \sum_{i=1}^n x_i^2 \end{bmatrix} \\
	&= \begin{bmatrix} n & n \bar{x} \\ n \bar{x} & n \overline{x^2} \end{bmatrix}
\end{aligned}
\end{equation*}
::::

:::

---

\begin{equation*}
\begin{aligned}
	X^T X &= \begin{bmatrix} n & n \bar{x} \\ n \bar{x} & n\overline{x^2} \end{bmatrix} \\
	(X^T X)^{-1} &= \frac{1}{n^2\overline{x^2} - n^2 \bar{x}^2}\begin{bmatrix} n \overline{x^2} & -n \bar{x} \\ -n \bar{x} & n \end{bmatrix} \\
	&= \frac{1}{n(\overline{x^2} - \bar{x}^2)}\begin{bmatrix} \overline{x^2} & -\bar{x} \\ -\bar{x} & 1 \end{bmatrix} \\
\end{aligned}
\end{equation*}

## 2d
\begin{equation*}
\begin{aligned}
	(X^T X)^{-1} X^T y &= \frac{1}{n(\overline{x^2} - \bar{x}^2)}\begin{bmatrix} \overline{x^2} & -\bar{x} \\ -\bar{x} & 1 \end{bmatrix} \begin{bmatrix} n \bar{y} \\ n \overline{xy} \end{bmatrix} \\
	&= \frac{1}{\overline{x^2} - \bar{x}^2}\begin{bmatrix} \overline{x^2}\bar{y} - \bar{x} \overline{xy} \\ \overline{xy} - \bar{x} \bar{y} \end{bmatrix} \\
	&= \begin{bmatrix} \bar{y} - \hat{w}_1 \bar{x} \\ \frac{\overline{xy} - \bar{x} \bar{y}}{\overline{x^2} - \bar{x}^2} \end{bmatrix}
\end{aligned}
\end{equation*}

# 2e - Jupyter Time
