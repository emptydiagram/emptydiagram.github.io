---
title: Backpropagation for an MLP classifier on MNIST
date: 2023 Dec 16
---

# Preliminaries

Below we use row-major convention to make the math match the implementation. All vectors are row vectors, which matches the usual data matrix $X$ being a vertical stack of training example row vectors.

For

$$\begin{bmatrix} s_1, \ldots, s_n \end{bmatrix} = \text{softmax}([x_1, \ldots, x_n]) = \begin{bmatrix} \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}} : i = 1, \ldots, n \end{bmatrix}$$

then the softmax partial derivatives are:

$$\frac{\partial s_j}{x_i} = s_j (\delta_{ij} - s_i)$$

where $\delta_{ij}$ is the Kronecker delta. Since the softmax Jacobian is symmetric, it is unchanged under row-major notation convention.

We use notation

$$\delta[k, :] := \begin{bmatrix} \delta_{k1} & \ldots & \delta_{kn} \end{bmatrix}$$

to denote a row vector implementing an indicator variable for the $k$-th component (all components are 0 save for the $k$-th, which is 1). The length $n$ is usually not explicitly notated, and is instead inferred from context.

The element-wise product between tensors of the same shape is the Hadamard product, denoted $\odot$.

The cross-entropy between the true random variable $X \sim p$ and an estimate $\hat{X} \sim q$, where $p$ and $q$ are finite probability distributions $p = [p_1, \ldots, p_n], q = [q_1, \ldots, q_n]$, is:

$$CE(p, q) := H(X, \hat{X}) := \mathbb{E}_{X \sim p}[- \log(\hat{X})] = - \sum_{i=1}^n p_i \log(q_i)$$

This is the average information content from estimating a distribution $q$ when the true distribution is $p$. Alternatively: the average negative log probability of our estimate of the outcomes, averaged with respect to the true probabilities.

We define, for $n \in \mathbb{N}$:

$$[n] := \{1, \ldots, n \}$$


# MLP classification

Let $d = 28^2 = 784$ be the input size, $h$ be the hidden size, $c = 10$ be the output size (number of classes), and $m$ be the minibatch size. Let $X \in \mathbb{R}^{m \times d}$ be a minibatch of inputs. Then the first layer pre-activation and activation are:

$$A^{(1)} = X W^{(1)} + b^{(1)}$$

$$Z^{(1)} = \phi(A^{(1)})$$

where $W^{(1)} \in \mathbb{R}^{d \times h}$ is the weight matrix of the fully-connected hidden layer, $b^{(1)} \in \mathbb{R}^h$ is the bias vector, and $\phi$ is a differentiable activation function $\phi$ applied component-wise. The addition in the pre-activation broadcasts the bias vector to all rows of $X W^{(1)}$.

The output layer is another linear layer ($W^{(2)} \in \mathbb{R}^{h \times c}, b^{(2)} \in \mathbb{R}^c$), followed by a softmax:

$$A^{(2)} = Z^{(1)} W^{(2)} + b^{(2)}$$

$$Z^{(2)} = \text{softmax}(A^{(2)})$$

As a result, $A^{(2)}, Z^{(2)} \in \mathbb{R}^{m \times c}$. Each row in $A^{(2)}_{[i,:]}$ is the *logits* vector for example $i$ in the minibatch, while $Z^{(2)}_{[i,:]}$  is the prediction for example $i$.

For each minibatch, there is also a ($m \times 1$) column vector of labels $y^\top$, where each label $y_i \in [c]$. One-hot encoding of each $y_i$ into a $1 \times c$ vector $[y_{i1}, \ldots, y_{ic}]$ turns it into a probability distribution over the $c$ classes, allowing us to apply cross-entropy between each logits vector $Z^{(2)}_i$ (the network's prediction of the class for each example) and the ground-truth label $y_i$.

We further define $Y := [y_{ij}]_{(i,j) \in [m] \times [c]}$,  i.e. the $i$-th row of $Y$ is $Y_{[i,:]} = \delta[y_i, :]$. Therefore we obtain the single-example loss:

$$\mathcal{L}(Z^{(2)}_{[i,:]}, y_i) := CE(Y_{[i,:]}, Z^{(2)}_{[i,:]}) = - \log(Z^{(2)}_{[i, y_i]})$$

Therefore, with respect to the minibatch $(X, y^\top)$, the average cross-entropy loss is:

$$\mathfrak{L}(Z^{(2)}, y^\top) = -\frac{1}{m} \sum_{i=1}^m \log(Z^{(2)}_{[i, y_i]})$$

which we want to minimize by adjusting the weights and biases.

# the best part was when he said "IT'S CALCULUS TIME" and calculused all over those guys

We wish to compute loss gradients $\frac{\partial \mathcal{L}}{\partial W^{(k)}}$, $\frac{\partial \mathcal{L}}{\partial b^{(k)}}$ for $k = 1, 2$. For simplicity, we begin by calculating partial derivatives for a single training example $(x, y) \in \mathbb{R}^d \times [c]$ (i.e. a minibatch of size 1). In this case, the corresponding pre-activations and activations for the input $x$: ($a^{(\ell)}, z^{(\ell)}$ for $\ell = 1, 2$) are row vectors. Also, the single-example loss $\mathcal{L}$ is defined by: $\mathcal{L} = - \log(z^{2}_y)$, so we have the immediate derivatives:

$$\frac{\partial a^{(1)}_k}{\partial W^{(1)}_{ij}} = \frac{\partial}{\partial W^{(1)}_{ij}} [x W^{(1)}_{[:,k]} + b^{(1)}_k] = \delta_{jk} x_i$$

$$\frac{\partial a^{(1)}_k}{\partial b^{(1)}_j} = \frac{\partial}{\partial b^{(1)}_j} [x W^{(1)}_{[:,k]} + b^{(1)}_k] = \delta_{jk}$$

$$\frac{\partial z^{(1)}_j}{\partial a^{(1)}_i} = \frac{\partial}{\partial a^{(1)}_i} [\phi(a^{(1)}_j)] = \delta_{ij} \phi'(a^{(1)}_j)$$

$$\frac{\partial a^{(2)}_j}{\partial z^{(1)}_i } = \frac{\partial}{\partial z^{(1)}_i} [z^{(1)} W^{(2)}_j + b^{(2)}_j] = W^{(2)}_{ij}$$

$$\frac{\partial a^{(2)}_k}{\partial W^{(2)}_{ij}} = \frac{\partial}{\partial W^{(2)}_{ij}} [z^{(1)} W^{(2)}_{[:,k]} + b^{(2)}_k] = \delta_{jk} z^{(1)}_i$$

$$\frac{\partial a^{(2)}_k}{\partial b^{(2)}_j} = \frac{\partial}{\partial b^{(2)}_j} [z^{(1)} W^{(2)}_{[:,k]} + b^{(2)}_k] = \delta_{jk}$$

$$\frac{\partial z^{(2)}_j}{\partial a^{(2)}_i} = \frac{\partial}{\partial a^{(2)}_i} [\text{softmax}(a^{(2)}_j)] = z^{(2)}_j (\delta_{ij} - z^{(2)}_i)$$

$$\frac{\partial \mathcal{L}(z^{(2)}, y)}{\partial z^{(2)}_i} = \frac{\partial}{\partial z^{(2)}_i} [-\log(z^{(2)}_y) ] = - \frac{\delta_{iy}}{z^{(2)}_y}$$

Using the chain rule (a.k.a. backpropagation along the network), we can put these together and obtain:

$$\frac{\partial \mathcal{L}}{\partial a^{(2)}_j} = \sum_{k \in [c]} \frac{\partial \mathcal{L}}{\partial z^{(2)}_k} \frac{\partial z^{(2)}_k}{\partial a^{(2)}_j} = - \sum_{k \in [c]} \frac{\delta_{ky}}{z^{(2)}_y} z^{(2)}_k (\delta_{jk} - z^{(2)}_j) = -(\delta_{yj} - z^{(2)}_j) $$

$$\frac{\partial \mathcal{L}}{\partial W^{(2)}_{ij}} = \frac{\partial \mathcal{L}}{\partial a^{(2)}_j} \frac{\partial a^{(2)}_j}{\partial W^{(2)}_{ij}} = -(\delta_{yj} - z^{(2)}_j) z^{(1)}_i$$

$$\frac{\partial \mathcal{L}}{\partial b^{(2)}_j} = \frac{\partial \mathcal{L}}{\partial a^{(2)}_j} \frac{\partial a^{(2)}_j}{\partial b^{(2)}_j} = -(\delta_{yj} - z^{(2)}_j)$$

$$\frac{\partial \mathcal{L}}{\partial z^{(1)}_i} = \sum_{j \in [c]} \frac{\partial \mathcal{L}}{\partial a^{(2)}_j} \frac{\partial a^{(2)}_j}{\partial z^{(1)}_i} = - \sum_{j \in [c]} (\delta_{yj} - z^{(2)}_j) W_{ij}^{(2)} = -(\delta[y,:] - z^{(2)}) \cdot W_{[i,:]}^{(2)}$$

$$\frac{\partial \mathcal{L}}{\partial a^{(1)}_j} = \frac{\partial \mathcal{L}}{\partial z^{(1)}_j} \frac{\partial z^{(1)}_j}{\partial a^{(1)}_j} = \frac{\partial \mathcal{L}}{\partial z^{(1)}_j} \cdot \phi'(a^{(1)}_j)$$

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}_{ij}} = \frac{\partial \mathcal{L}}{\partial a^{(1)}_j} \frac{\partial a^{(1)}_j}{\partial W^{(1)}_{ij}} = \frac{\partial \mathcal{L}}{\partial a^{(1)}_j} x_i$$

$$\frac{\partial \mathcal{L}}{\partial b^{(1)}_j} = \frac{\partial \mathcal{L}}{\partial a^{(1)}_j} \frac{\partial a^{(1)}_j}{\partial b^{(1)}_j} = \frac{\partial \mathcal{L}}{\partial a^{(1)}_j}$$

To vectorize, we can use:

$$\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \frac{\partial \mathcal{L}}{\partial a^{(2)}} = -(\delta[y, :] - z^{(2)})$$

$$\frac{\partial \mathcal{L}}{\partial W^{(2)}} = (z^{(1)})^\top \frac{\partial \mathcal{L}}{\partial a^{(2)}} = - (z^{(1)})^\top (\delta[y, :] - z^{(2)})$$

$$\frac{\partial \mathcal{L}}{\partial z^{(1)}} = - (\delta[y, :] - z^{(2)}) W^{(2) \top} = \frac{\partial \mathcal{L}}{\partial a^{(2)}} W^{(2) \top} $$

$$\frac{\partial \mathcal{L}}{\partial b^{(1)}} = \frac{\partial \mathcal{L}}{\partial a^{(1)}} = \frac{\partial \mathcal{L}}{\partial z^{(1)}} \odot \phi'(a^{(1)})$$

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = x^\top \cdot \frac{\partial \mathcal{L}}{\partial a^{(1)}}$$

The above is for a single training example. For a minibatch $(X, y^\top) \in \mathbb{R}^{m \times d} \times [c]^{m \times 1}$ of examples, we can average over the examples, so that, if we define $\mathcal{L}_k$ to be the single example loss for the $k$-th training sample in the batch, we get:

$$\mathfrak{L} = \frac{1}{m} \sum_{k=1}^m \mathcal{L}_k$$

Then:

$$\frac{\partial \mathfrak{L}}{\partial A^{(2)}_{[i,:]}} = \frac{1}{m} \sum_{k=1}^m \frac{\partial \mathcal{L}_k}{\partial A^{(2)}_{[i, :]}} = \frac{1}{m} \frac{\partial \mathcal{L}_i}{\partial A^{(2)}_{[i, :]}} = - \frac{1}{m} (Y_{[i,:]} - Z^{(2)}_{[i,:]})$$


$$\frac{\partial \mathfrak{L}}{\partial A^{(2)}} = - \frac{1}{m} (Y - Z^{(2)})$$

$$\frac{\partial \mathfrak{L}}{\partial b^{(2)}} = \frac{1}{m} \sum_{i=1}^m \frac{\partial \mathcal{L}_i}{\partial A^{(2)}_{[i,:]}} = - \frac{1}{m} \sum_{i=1}^m (Y_{[i, :]} - Z^{(2)}_{[i, :]}) = \text{average along rows of } -(Y - Z^{(2)})$$


$$\frac{\partial \mathfrak{L}}{\partial W^{(2)}} = - \frac{1}{m} \sum_{i=1}^m (Z^{(1)}_{[i,:]})^\top (Y_{[i, :]} - Z^{(2)}_{[i, :]}) = Z^{(1)\top} \frac{\partial \mathfrak{L}}{\partial A^{(2)}}$$


$$\frac{\partial \mathfrak{L}}{\partial Z^{(1)}_{[i,:]}} = \frac{1}{m} \sum_{k=1}^m \frac{\partial \mathcal{L}_k}{\partial Z^{(1)}_{[i,:]}} = - \frac{1}{m} (Y_{[i,:]} - Z^{(2)}_{[i,:]}) W^{(2) \top} = \frac{\partial \mathfrak{L}}{\partial A^{(2)}_{[i,:]}} W^{(2) \top} $$

$$\frac{\partial \mathfrak{L}}{\partial Z^{(1)}} = \frac{\partial \mathfrak{L}}{\partial A^{(2)}} W^{(2) \top} $$

$$\frac{\partial \mathfrak{L}}{\partial A^{(1)}_{[i,:]}} = \frac{1}{m} \sum_{k=1}^m \frac{\partial \mathcal{L}_k}{\partial A^{(1)}_{[i, :]}} = \frac{1}{m} \frac{\partial \mathcal{L}_i}{\partial A^{(1)}_{[i, :]}} = \frac{1}{m} \frac{\partial \mathcal{L}_i}{\partial Z^{(1)}_{[i, :]}} \odot \phi'(A^{(1)}_{[i, :]}) = \frac{1}{m} \frac{\partial \mathfrak{L}}{\partial Z^{(1)}_{[i, :]}} \odot \phi'(A^{(1)}_{[i, :]})$$

$$\frac{\partial \mathfrak{L}}{\partial A^{(1)}} = \frac{1}{m} \frac{\partial \mathfrak{L}}{\partial Z^{(1)}} \odot \phi'(A^{(1)})$$

$$\frac{\partial \mathfrak{L}}{\partial b^{(1)}} = \frac{1}{m} \sum_{i=1}^m \frac{\partial \mathcal{L}_i}{\partial b^{(1)}} = \frac{1}{m} \sum_{i=1}^m \frac{\partial \mathcal{L}_i}{\partial Z^{(1)}_{[i,:]}} \odot \phi'(A^{(1)}_{[i,:]}) = \text{average along rows of } (\frac{\partial \mathfrak{L}}{\partial Z^{(2)}} \odot \phi'(A^{(1)}))$$

$$\frac{\partial \mathfrak{L}}{\partial W^{(1)}} = \frac{1}{m} \sum_{k=1}^m \frac{\partial \mathcal{L}_k}{\partial W^{(1)}} = \frac{1}{m} \sum_{k=1}^m (X_{[k,:]})^\top \frac{\partial \mathcal{L}_k}{\partial A^{(1)}_{[k,:]}} = \frac{1}{m} X^\top \frac{\partial \mathfrak{L}}{\partial A^{(1)}}$$


