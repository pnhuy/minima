---
layout: page
title: Fundamentals of Unconstrained Optimization
subtitle: Notes on Numerical Optimization
categories: optimization
math: true
---

In unconstrained optimization, we minimize an objective function that depends on real variable, with no restriction:

$$\min_{x} f(x)$$

where $x \in \mathbb{R}^n$ is a real vector with $n \ge 1$ components and $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is a smooth function.

## What is a solution?
{:.label}

<div class="definition">
A point $x*$ is a <em>global minimizer</em> if $f(x^*) \le f(x), \forall x \in \mathbb{R}^n$
</div>

The global minimizer can be difficult to find. And most algorithms are able to find only a *local minimizer*:

<div class="definition">
A point $x*$ is a <em>local minimizer</em> if there is a neighborhood N of $x*$ such that $f(x^*) \le f(x), \forall x \in N$
</div>

This is sometimes called a *weak local minimizer*.

<div class="definition">
A point $x*$ is a <em>strict local minimizer</em> (also called a strong local minimizer) if there is a neighborhood N of $x*$ such that $f(x^ *) < f(x), \forall x \in N$  with $x \ne x^ *$.
</div>

A slightly more exotic type of <em>local minimizer</em>:


<div class="definition">A point $x*$ is an <em>isolated local minimizer</em> if there is a neighborhood N of $x*$ such that $x*$ is the only local minimizer in N.</div>


While strict local minimizers are not always isolated, it is true that all isolated local minimizers are strict.

## Recognizing a local minimum

Is x* is a local minimum? → Examine all the points in its immediate vicinity to make sure that none of them has a smaller function value.

If f is twice continuously differentiable, we may be able to tell that x* is a local minimizer (and possibly a strict local minimizer) by examining just the gradient $\nabla f(x^ * )$ and the Hessian $\nabla^2 f(x^ * )$  by using Taylor's theorem.

<div class="theorem">
<strong>Taylor's Theorem</strong>
<br>
Suppose that $f: \mathbb{R}^2 \rightarrow \mathbb{R}$ is continuously differentiable and that $p \in \mathbb{R}^n$. Then we have that
$$f(x+p) = f(x) + \nabla f(x+tp)^Tp$$
for some $t \in (0,1)$.
<br>
Moreover, if $f$ is twice continuously differentiable, we have that 
$$\nabla f(x+p) = \nabla f(x) + \int_0^1 \nabla ^2 f(x+tp)p \mathrm{d}t$$
and that
$$f(x+p) = f(x) + \nabla f(x+tp)^Tp +\frac{1}{2}p^T \nabla ^2 f(x+tp)p$$
</div>

*Necessary conditions* for optimality are derived by assuming that x* is a local minimizer and then providing facts about $\nabla f(x^ * $ and $\nabla^2 f(x^ * )$.

<div class="theorem">
	<strong>First-Order Necessary Conditions</strong>
	<br>
	If x* is a local minimizer and f is continuous differentiable in an open neighborhood of x* , then $\nabla f(x^*) = 0$.
</div>

<div class="theorem">
	<strong>Second-Order Necessary Conditions</strong>
	<br>
	If x* is a local minimizer of $f$ and $\nabla^2 f$ is continuous in an open neighborhood of x* , then  $\nabla f(x^ * ) = 0$ and $\nabla^2 f(x^*)$  is positive semidefinite.
</div>

<div class="theorem">
	<strong>Second-Order Sufficient Conditions</strong>
	<br>
	Suppose that $\nabla^2 f$ is continuous in an open neighborhood of x* and that $\nabla f(x^ * ) = 0$ and $\nabla^2 f(x^ * ) $ is positive definite. Then x* is a strict *local minimizer* of $f$.
</div>

<div class="theorem">When f is convex, any local minimizer x* is a global minimizer of f. If in addition f is differentiable, then any stationary point x* is a global minimizer of f.</div>

## Algorithms

### Line Search *vs* Trust Region

General approach of algorithms for unconstrained minimization:

* Supply a starting point $x_0$
* Generate a sequence of iterates $\{x_k\}_{k=0}^\infty$
* Terminate when either no more progress can be made or when it seems that a solution point has been approximated with sufficient accuracy.

There are 2 common strategies for moving from the current point $x_k$ to a new iterate $x_{k+1}$: *Line search* and *Trust region*.

#### *Line search* strategy

* Choose a direction $p_k$ 
* Search along this direction from the current state $x_k$ for a new state with a lower function value
* The distance can be found by solving approximately ($\alpha$ is step length): 

$$
\min_{\alpha>0} f(x_k + \alpha p_k)
$$

The exact minimization is expensive and unnecessary. In fact, the line search algorithm generates a limited number of trial step lengths until it finds one that the loosely approximates the minimum of above minimization.

#### *Trust region* strategy

* Construct a *model function* $m_k$ that approximately similar to $f$ in neighborhood of $x_k$.
* Search minimizer of $m_k$ (also $x_k$) in some region around $x_k$. In other words, we find the candidate step p by approximately solving the following subproblem: 
  
  $$\min_{p} m_k(x_k+p)$$ 
  
  where $x_k +p$ lies inside the trust region.

* If solution does not produce a sufficient decrease in $f$ → *Trust region* is too large → Shrink it and re-solve.

  The trust region is defined by $\|p\|_2 \le \Delta$ where the scalar $\Delta > 0$ is called the trust-region radius. The model $m_k$  is usually defined to be a quadratic function of the form:

  $$m_k(x_k + p)= f_k + p^T \nabla f_k + \frac{1}{2} p^T B_k p$$

  where $f_k, \nabla f_k, B_k$ are a scalar, vector, matrix, respectively. $B_k$ is either the Hessian $\nabla^2 f_k$ or some approximation to it.

The *line search* and *trust region* approaches differ in the order in which they choose the direction and distance of the move to the next iterate.

For line search, fixing the direction $p_k$ → identifying an appropriate distance $\alpha_k$

For trust region, choose a maximum distance - the trust region radius $\Delta_k$ → seek a direction and step

#### Search directions for *Line search* methods

##### The steepest-descent direction

The most obvious choice for search direction for line search method is *the steepest-descent direction* $- \nabla f_k$. That means among all the directions we could move from $x_k$, this is the way that $f$ decreases most rapidly.

Let's prove this. By Taylor's theorem, we have

$$
f(x_k +\alpha p) = f(x_k) +\alpha p^T \nabla f_k + \frac{1}{2} \alpha^2 p^T \nabla^2 f(x_k + tp) p \text{, for some }t \in (o, \alpha)
$$

which p is the search direction and $\alpha$ is the step-length parameter.

So, in this formula, *the rate of change* in $f$ along the direction $p$ at $x_k$ is $p^T \nabla f_k$.

As the result, the unit direction $p$ of the most rapid decrease is the solution to the problem

$$
\min_p p^T \nabla f_k \text{, subject to } \|p\| =1
$$

Because $p^T \nabla f_k = \|p\| \| \nabla f_k \| \cos \theta$ and $\| p\|=1$, we have $p^T \nabla f_k = \| \nabla f_k \| \cos \theta$. Minimization in (9) occurs when $\cos \theta = -1$ at $\theta = \pi$. In other words, the solution to (9) is

$$
p = -\nabla f_k / \|\nabla f_k\|
$$

In visual representation, this direction is orthogonal to the contours of the function.

![The direction](http://trond.hjorteland.com/thesis/img200.gif)

**Advantage of the steepest descent direction** is that it require calculation of the gradient $\nabla f_k$ but not of second derivatives. However, it can be excruciatingly slow on difficult problems.

When the $\theta$ is not exactly $\pi$ or the direction is just descent, Taylor's theorem also verifies that $f$ is still decreased with a sufficiently small. In this situation, $p_k$ is *a down hill direction*.

##### Newton direction

From the second-order Taylor series approximation to $f(x_k + p)$ :

$$
f(x_k + p) \approx f_k + p^T \nabla f_k + \frac{1}{2} p^T \nabla^2 f_k p \overset{def}{=} m_k(p)
$$

Assuming that $\nabla^2 f_k$ is positive definite, the Newton direction is defined by finding the vector $p$ that minimizes $m_k (p)$. By simply setting the derivative of $m_k(p)$ to zero, we obtain the following explicit formula:

$$
p^N_k = -\nabla^2 f_k^{-1} \nabla f_k
$$

**When $\nabla^2 f_k$ is positive definite**, we have

$$
\nabla f_k^T p_k^N = -p_k^{NT} \nabla^2 f_k p_k^N \le -\sigma_k \| p_k^N \|^2 \text{, for some }\sigma_k > 0
$$

Unless the gradient $\nabla f_k$ is zero, we have that $\nabla f_k^T p_k^N < 0 $, so the Newton direction is a descent direction. 

Unlike the steepest descent direction, there is a "natural" step length of 1 associated with the Newton direction. Most line search implementations of Newton's method use the unit step $\alpha = 1$ where possible and adjust this step length only when it does not produce a satisfactory reduction in the value of $f$.

**When** $\nabla^2 f_k$ **is not positive definite**, the Newton direction may not even be defined, since $\nabla^2 f_k^{-1}$ may not exist. Even when it is defined, it may not satisfy the descent property $\nabla f_k^T p_k^N  < 0$. So that, it is unsuitable as a search direction.

In these cases,  *$p_k$ need to be changed* to guarantee the downhill condition while keeping the benefit of second-order information in $\nabla^2 f_k$.

The Newton direction approach has a fast rate of local convergence - quadratic.

*The main disadvantage* of the Newton direction is the need for the Hessian $\nabla^2 f(x)$. This is usually a cumbersome, error-prone and expensive process.

##### Quasi-Newton direction

This method do not require computation of the Hessian but still have a super linear rate of convergence.

They use an approximation $B_k$ for the true Hessian $\nabla^2 f_k$. In this case, $B_k$ is updated to take account of the additional knowledge gained after each step. The updates make use of the fact that changes in the gradient $g$ provide information about the second derivative of $f$ along the search direction.

\\[
\nabla f(x+p) = \nabla f(x) + \nabla^2 f(x) p + \int_{0}^{1} \left[\nabla^2 f(x+tp) - \nabla^2 f(x) \right]p dt
\\]

Because $\nabla f(.)$ is continuous, the size of final integral term is $o(\|p\|)$. By setting $x = x_k$ and $p = x_{k+1} - x_k$:
\\[
\nabla f_{k+1} = \nabla f_k + \nabla^2 f_{k+1} (x_{k+1} - x_k) + o(\| x_{k+1} - x_k \|)
\\]

When $x_k$ and $x_{k+1}$ lie nearly the solution $x*$, within which $\nabla f$ is positive definite, the final term in this expansion is eventually dominated by the $\nabla^2 f_k (x_{k+1} - x_k)$ and $\nabla^2 f_{k+1} (x_{k+1} - x_k) \approx \nabla f_{k+1} - \nabla f_k$

We choose the new Hessian approximation $B_{k+1}$ so that it mimics this property. This means that it needs to satisfy the *secant equation*:
\\[
B_{k+1} s_k = y_k
\\]
where
\\[
s_k = x_{k+1} - x_k, y_k = \nabla f_{k+1} - \nabla f_k
\\]

There are two most popular way to choose $B_{k+1}$:
* The *symmetric-rank-one* (SR1) formula:
	\\[
	B_{k+1} = B_k + \frac{(y_k - B_k s_k)(y_k - B_k s_k)^T}{(y_k - B_k s_k)^T s_k}
	\\]
* The *BFGS* (Broyden-Fletcher-Goldfarb-Shanno) formula (which is a rank-two matrix):
	\\[
	B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{y_k y_k^T}{y_k^T s_k}
	\\]

The quasi-Newton search direction is $p_k = -B_k^{-1} \nabla f_k$

By applying the inverse approximation $H_k \overset{def}{=} B_k^{-1}$, the above formula is equivalent to
\\[
H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T, \rho_k = \frac{1}{y_k^T s_k}
\\]

This can be implemented as a matrix-vector multiplication, which is typically simpler than the previous formula.

##### Nonlinear conjugate gradient methods

This method of search direction has form
\\[
p_k = -\nabla f(x_k) + \beta_k p_{k-1}
\\]

where $\beta_k$ is a scalar that ensure that $p_k$ and $p_{k-1}$ are <em>conjugate</em>.\\
This method is much more effective than the steepest descent direction, not requiring storage of matrices and simple to compute but does not retain the fast convergence rates of Newton or quasi-Newton methods.

#### Models for Trust-region methods

The subproblem of Trust-region is

$$\min_{p} m_k(x_k + p)$$

where $x_k + p$ lies inside the trust region.

The model function has the quadratic fuction form:

$$m_k(x_k + p)= f_k + p^T \nabla f_k + \frac{1}{2} p^T B_k p$$

We have some ways to choose $B_k$:
* We set $B_k = 0$ and define the trust region using the Euclidean norm, the trust region subproblem becomes
	\\[
	\min_{p} f_k + p^T \nabla f_k \text{, subject to }\| p \|_2 \le \Delta_k
	\\]
	We can write the solution to this problem in closed form as
	\\[
	p_k = -\frac{\Delta_k \nabla f_k}{\| \nabla f_k \|}
	\\]
	This is simply a steepest descent step in which the step length is determined by the trust-region radius; the trust-region and line search approaches are essentially the same in this case.
* We can choose $B_k$ to be exact Hessian $\nabla^2 f_k$ in quadratic model. The subprolem is guaranteed to have a solution $p_k$ even though the $\nabla^2 f_k$ is not positive definite because of the restriction $\| p\|_2 \le \Delta_k$.
* We are also able to choose $B_k$ by means of a quasi-Newton approximation. This is *trust-region quasi-Newton method*.


## Scaling

<!-- \begin{figure}
	\centering
	\includegraphics[width=0.5\linewidth]{scaling.png}
	\caption{Poorly scaled and Well-scaled problems. In the poorly scaled problem, the steepest descent direction does not yield much reduction in the function, while the well-scaled problem performs much better.}
	\label{fig:scaling}
\end{figure} -->

A uncontrained optimization is *poorly scaled* if changes to x in a certain direction prodct much larger variations in the value of $f$ than do changes to x in another direction.

Steepest descent and som optimization algorithms are sensitive to poor scaling, while others such as Newton's method are unaffected by it.
In design complete algorithms, *scale invariance* is included in all aspects of algoritm. In general, it is easier to preserve scale invariance for line search algorithms than for trust-region algorithms.

## Rates of convergence

<div class="definition">
	Let $\{x_k\}$ be a sequence in $\mathbb{R}^n$ that converges to $x^{*}$.
	We say that the convergence is <strong>Q-linear</strong> if there is a constant $r \in (0,1)$ such that
	$$\frac{\| x_{k+1} - x* \|}{\| x_k - x* \|} \le r \text{, for all k sufficiently large}$$
	The convergence is said to be <strong>Q-superlinear</strong> if
	$$\lim_{k \rightarrow \infty} \frac{\| x_{k+1} - x* \|}{\| x_k - x* \|} = 0 \text{, for all k sufficiently large}$$
	<strong>Q-quadratic</strong> is optained if
	$$\frac{\| x_{k+1} - x* \|}{\| x_k - x* \|^2} \le M \text{, for all k sufficiently large}$$
	In general, the <strong>Q-order</strong> of convergence is p (with p > 1) if there is a positive constant M such that
	$$\frac{\| x_{k+1} - x* \|}{\| x_k - x* \|^p} \le M \text{, for all k sufficiently large}$$
</div>

For example:
* $1+(0.5)^k$ converges *Q-linearly* to 1
* $1+k^{-k}$ converges *superlinearly* to 1
* $1+ (0.5)^{2^k}$ converses *quadratically* to 1

The speed of convergence depends on $r$ and $M$. These values depend on both algorithms and properties of problem. However, a quadratically convergent sequence will always eventually converge faster than a linearly convergent sequence.

Quasi-Newton methods typically converge Q-superlinearly, whereas Newton's method converges Q-quadratically. In contrast, steepest descent algorithms converge only at a Q-linear rate, and when the problem is ill-conditioned the convergence constant $r$ is close to 1.

