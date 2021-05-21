---
layout: page
title: Introduction to Graph Theory
subtitle: Notes from the book Discrete Mathematics with Application (Rosen)
math: true
---

## Graph and Graph models

A graph $G=(V,E)$ consists of $V$, an nonempty set of *vertices* (or *nodes*) and $E$, a set of edges. Each edge has either one or two vertices associated with it, called its *endpoints*. An edge is said to *connect* its endpoints.

A graph with infinite vertex or egdes is called **infinite graph**. In the otherwise case, **finite graph**.

A graph in which each edge connects two different vertices and where no two edges connect the same pair of vertices is called a **simple graph**.

Graph that may have multiple edges connecting the same vertices are called **multigraphs**.

The edges that connect a vertex to itself are called **loops**.

Graphs including loops, and possibly multiple edges connecting the same pair of vertices or a vertex to itself, are called **pseudographs**.

The graph with **undirected** edges is an **undirected graph**.

A directed graph (or digraph) $(V,E)$ consists of a nonempty set of vertices $V$ and a set of directed edges (or arcs) $E$. Each directed edge is associated with an ordered pair of vertices. The directed edge associated with the ordered pair $(u,v)$ is said to start at $u$ and end at $v$.

A directed graph has no loops and has no multiple directed edges is called a **simple directed graph**.

Directed graphs that may have **multiple directed edges** from a vertex to a second (possibly the same) vertex are used to model such networks are called **directed multigraphs**. When there are $m$ directed edges, each associated to an ordered pair of vertices $(u,v)$, we say that $(u,v)$ is an edge of multiplicity $m$.

A graph with both directed and undirected edges is called a **mixed graph**.

Graphs are used in a wide variety of models:
-   Social networks
-   Communication networks
-   Information networks
-   Software design applications
-   Transportation networks
-   Biological networks
-   Tournaments

## Terminology and special types of graph

### Terminology

Two vertices u and v in an undirected graph G are called **adjacent** (or **neighbors**) in G if u and v are endpoints of an edge e of G. Such an edge e is called **incident with** the vertices u and v and e is said to connect u and v.

The set of all neighbors of a vertex v of $G=(V,E)$, denoted by $N(v)$, is called the **neighborhood** of $v$. If $A$ is a subset of $V$ , we denote by $N(A)$ the set of all vertices in $G$ that are adjacent to at least one vertex in A. So, $N(A)=\bigcup_{v\in A}N(v)$.

The **degree of a vertex in an undirected graph** is the number of edges incident with it, *except that a loop at a vertex contributes twice to the degree of that vertex*. The degree of the vertex v is denoted by $deg(v)$.

A vertex of degree zero is called **isolated** (not adjacent to any vertex).

A vertex is **pendant** if and only if it has degree one (adjacent to exactly one other vertex).

<div class="theorem">
The handshaking theorem <br>
Let $G=(V,E)$ be an undirected graph with $m$ edges:
$$2m=\sum_{v\in V}\deg(v)$$
</div>

This theorem shows that the sum of the degrees of the vertices of an undirected graph is even, which lead to the below theorem.

<div class="theorem">An undirected graph has an even number of vertices of odd degree.
</div>

When $(u,v)$ is an edge of the graph G with directed edges, $u$ is said to be **adjacent to** $v$ and $v$ is said to be **adjacent from** $u$. The vertex $u$ is called the **initial vertex** of $(u,v)$, and $v$ is called the **terminal** or **end vertex** of $(u,v)$. The initial vertex and terminal vertex of a loop are the same.

In a graph with directed edges the **in-degree** of a vertex v, denoted by $\deg^{-}(v)$, is the number of edges with v as their terminal vertex. The **out-degree** of v, denoted by $\deg^{+}(v)$, is the number of edges with v as their initial vertex. (Note that a loop at a vertex contributes 1 to both the in-degree and the out-degree of this vertex.)

<div class="theorem">
Let $G=(V,E)$be a graph with directed edges:
$$\sum_{v\in V}\deg^{-}(v)=\sum_{v\in V}\deg^{+}(v)=|E|$$
</div>

Some special simple graphs:
* Complete Graphs: a complete graph on n vertices, denoted by $K_{n}$, is a simple graph that contains exactly one edge between each pair of distinct vertices.
* Wheels
* n-cube / n-dimensional hypercube, denoted by $Q_{n}$, is a graph that has vertices representing the $2^{n}$ bit strings of length n.

### Bipartite Graph

A simple graph $G$ is called **bipartite** if its vertex set $V$ can be partitioned into two disjoint sets $V_{1}$ and $V_{2}$ such that every edge in the graph connects a vertex in $V_{1}$ and a vertex in $V_{2}$ (so that no edge in $G$ connects either two vertices in $V_{1}$ or two vertices in $V_{2}$). When this condition holds, we call the pair $(V_{1},V_{2})$ a bipartition of the vertex set $V$ of $G$.

<div class="theorem">
A simple graph is bipartite if and only if it is possible to assign one of two different colors to each vertex of the graph so that no two adjacent vertices are assigned the same color.
</div>

A **complete bipartite graph** $K_{m,n}$ is a graph that has its vertex set partitioned into two subsets of $m$ and $n$ vertices, respectively with an edge between two vertices if and only if one vertex is in the first subset and the other vertex is in the second subset.

A **matching** $M$ in a simple graph $G=(V,E)$ is a subset of the set $E$ of edges of the graph such that no two edges are incident with the same vertex. In other words, a matching is a subset of edges such that if ${s,t}$ and ${u,v}$ are distinct edges of the matching, then s, t, u, and v are distinct.

A vertex that is the endpoint of an edge of a matching $M$ is said to be matched in $M$; otherwise it is said to be unmatched.

A **maximum matching** is a matching with the largest number of edges. We say that a matching M in a bipartite graph $G=(V,E)$ with bipartition $(V_{1},V_{2})$ is a complete matching from $V_{1}$ to $V_{2}$ if every vertex in $V_{1}$ is the endpoint of an edge in the matching, or equivalently, if $\|M\|=\|V_1\|$.

<div class="theorem">
_(Hall's marriage theorem / Necessary and sufficient conditions for complete matching)_\\
The bipartite graph $G=(V,E)$ with bipartition $(V_{1},V_{2})$ has a complete matching from $V_{1}$ to $V_{2}$ if and only if $\|N(A)\| \ge \|A\|$ for all subsets A of $V_{1}$.
</div>

## Representing Graphs and Graph Isomorphism

### Adjacency Lists

One way to represent a graph without multiple edges is to list all the edges of this graph. Another way to represent a graph with no multiple edges is to use **adjacency lists**, which specify the vertices that are adjacent to each vertex of the graph.

![Adjacency List](https://storage.googleapis.com/algodailyrandomassets/curriculum/graphs/implementing-graphs-adjacencylist.png)

### Adjacency Matrices

Suppose that $G = (V , E)$ is a simple graph where $\|V\| = n$. Suppose that the vertices of $G$ are listed arbitrarily as $v_1, v_2, ..., v_n$ . 

The adjacency matrix $A$ (or $A_G$ ) of $G$, with respect to this listing of the vertices, is the $n \times n$ zero–one matrix with 1 as its $(i, j)$th entry when $v_i$ and $v_j$ are adjacent, and 0 as its $(i,j)$th entry when they are not adjacent. 

In other words, if its adjacency matrix is $A = [a_{ij}]$, then 

$$ 
a_{ij} = \left\{
\begin{matrix}
1 & \text{if } {v_i, v_j} \text{ is an edge of G} \\
0 & \text{otherwise}
\end{matrix}
\right.
$$

![Adjacency Matrix](https://storage.googleapis.com/algodailyrandomassets/curriculum/graphs/implementing-graphs-adjacencymatrix.png)

### Incidence Matrices

Let $G = (V , E)$ be an undirected graph. Suppose that $v_1 , v_2 , . . . , v_n$ are the vertices and $e1 , e2 , . . . , e_m$ are the edges of G. Then the incidence matrix with respect to this ordering of $V$ and $E$ is the $n \times m$ matrix $M = [m_{ij}]$, where

$$ 
m_{ij} = \left\{
\begin{matrix}
1 & \text{when edge } e_j \text{ is incident with } v_i  \\
0 & \text{otherwise}
\end{matrix}
\right.
$$

![Incidence Matrix](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Incidence_matrix_-_undirected_graph.svg/800px-Incidence_matrix_-_undirected_graph.svg.png)

### Isomorphism

<div class="definition">

The simple graphs $G_1 = (V_1 , E_1 )$ and $G_2 = (V_2 , E_2 )$ are isomorphic if there exists a one-to-one and onto function $f$ from $V_1$ to $V_2$ with the property that $a$ and $b$ are adjacent in $G_1$ if and only if $f(a)$ and $f(b)$ are adjacent in $G_2$ , for all $a$ and $b$ in $V_1$ . Such a function $f$ is called
an <strong>isomorphism</strong>. Two simple graphs that are not isomorphic are called <strong>nonisomorphic</strong>.

</div>

In other words, when two simple graphs are isomorphic, there is a one-to-one correspondence between vertices of the two graphs that preserves the adjacency relationship. Isomorphism of simple graphs is an equivalence relation.

<p align="center"><img src="https://qph.fs.quoracdn.net/main-qimg-16c40327f5a5eec560705ddf2e8eceb9"></p>

A property preserved by isomorphism of graphs is called a **graph invariant**. For example, isomorpic graphs must have:
* The same # of vertices
* The same # of edges
* The degrees of the vertices in the graphs must be the same

## Connectivity
### Paths
{:.label}

**Definition**{:.label}\\
Let $n$ be a nonnegative integer and $G$ an undirected graph. \\
A **path** of length $n$ from $u$ to $v$ in $G$ is a sequence of $n$ edges $e_1,...,e_n$ of $G$ for which there exists a sequence $x_0 = u,x_1,\dots,x_{n−1},x_n = v$ of vertices such that $e_i$ has, for $i = 1,\dots,n$, the endpoints $x_{i−1}$ and $x_i$.
{:.definition}

When the graph is simple, we denote this path by its vertex sequence $x_0,x_1,\dots,x_n$ (because listing these vertices uniquely determines the path).

The path is a circuit if it begins and ends at the same vertex, that is, if u = v, and has length greater than zero.

The path or circuit is said to pass through the vertices $x_1, x_2,\dots, x_{n−1}$ or traverse the edges $e_1, e_2,\dots, e_n$. A path or circuit is simple if it does not contain the same edge more than once.

Read more: [Erdős Number](https://mathworld.wolfram.com/ErdosNumber.html), [Bacon number](https://simple.wikipedia.org/wiki/Bacon_number).

### Connectedness in Undirected Graphs
{:.label}

**Definition**{:.label}\\
An **undirected graph** is called *connected* if there is a path between every pair of distinct
vertices of the graph.\\
An **undirected graph** that is *not connected* is called *disconnected*.\\
We say that we disconnect a graph when we remove vertices or edges, or both, to produce a
disconnected subgraph.
{:.definition}

<figure>
  <img src="https://walkenho.github.io/images/graph-theory-and-networkX-part2-fig1.jpg" align="center">
  <figcaption>Connected vs Disconnected Graph (Credit <a href="https://walkenho.github.io/graph-theory-and-networkX-part2/">walkenho</a>)</figcaption>
</figure>

**Theorem**{:.label}
There is a simple path between every pair of distinct vertices of a connected undirected graph.
{:.theorem}


**Definition**{:.label}\\
A **connected component** of a graph G is a connected subgraph of G that is not a proper subgraph of another connected subgraph of G.
{:.definition}

That is, a connected component of a graph G is a maximal connected subgraph of G.

A graph G that is not connected has two or more connected components that are disjoint and have G as their union.

<figure>
  <img src="https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/46457/versions/3/screenshot.jpg" width="75%">
  <figcaption>Connected Components (Credit <a href="https://www.mathworks.com/matlabcentral/fileexchange/46457-splitting-a-network-into-connected-components">mathworks</a>)</figcaption>
</figure>

Read more: [Finding Connected Components using DFS](https://www.baeldung.com/cs/graph-connected-components#finding-connected-components)

### How Connected is a Graph?
{:.label}

**Definition**{:.label}\\
**Cut vertices** or **Articulation points** is the vertices that if they and their incident edges are removed from a graph, a subgraph with more connected components is produced.\\
An edge whose removal produces a graph with more connected components than in the original graph is called a **Cut edge** or **Bridge**.
{:.definition}

<figure>
    <img src="https://static.javatpoint.com/tutorial/graph-theory/images/connectivity3.png">
    <figcaption>The vertices b, c, e is cut vertices and the edge (c, e) is cut edge <br> (Credit <a href="https://www.javatpoint.com/graph-theory-connectivity">javatpoint</a>)</figcaption>
</figure>

#### Vertex Connectivity

When a complete graph $K_n$ removes a vertex and all incidents to it, the resulting graph is a complete graph $K_{n-1}$ which also a connected graph.

<figure>
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/kargest-subset-of-graph-vertices-with-edges-of-2-or-more-colors-3.png">
    <figcaption>Remove a vertex from a $K_6$ makes a $K_5$ graph (Credit <a href="https://www.geeksforgeeks.org/largest-subset-graph-vertices-edges-2-colors/">geeksforgeeks</a>)</figcaption>
</figure>

Connected graph having no cut vertex is a **nonseparable graph** which is more connected than a graph with a cut vertex.

**Definition**{:.label}\\
A subset $V'$ of the vertex set $V$ of $G=(V,E)$ is a **vertex cut**, or **separating set**, if $G-V'$ is disconnected.
{:.definition}

#### Edge Connectivity




### Connectedness in Directed Graphs
{:.label}

### Paths and Isomorphism
{:.label}

### Counting Paths Between Vertices
{:.label}

## Euler and Hamilton Paths
{:.label}

### Euler Paths and Circuits
{:.label}

### Hamilton Paths and Circuits
{:.label}


