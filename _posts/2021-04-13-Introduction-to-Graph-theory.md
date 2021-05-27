---
layout: page
title: Introduction to Graph Theory
subtitle: Notes from the book Discrete Mathematics with Application (Kenneth H. Rosen)
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

<div class="theorem">
    An undirected graph has an even number of vertices of odd degree.
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
<strong>Hall's marriage theorem / Necessary and sufficient conditions for complete matching</strong>
<br>
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

<div class="definition">
Let $n$ be a nonnegative integer and $G$ an undirected graph.
<br>
A <strong>path</strong> of length $n$ from $u$ to $v$ in $G$ is a sequence of $n$ edges $e_1,...,e_n$ of $G$ for which there exists a sequence $x_0 = u,x_1,\dots,x_{n−1},x_n = v$ of vertices such that $e_i$ has, for $i = 1,\dots,n$, the endpoints $x_{i−1}$ and $x_i$.
</div>

When the graph is simple, we denote this path by its vertex sequence $x_0,x_1,\dots,x_n$ (because listing these vertices uniquely determines the path).

The path is a circuit if it begins and ends at the same vertex, that is, if u = v, and has length greater than zero.

The path or circuit is said to pass through the vertices $x_1, x_2,\dots, x_{n−1}$ or traverse the edges $e_1, e_2,\dots, e_n$. A path or circuit is simple if it does not contain the same edge more than once.

Read more: [Erdős Number](https://mathworld.wolfram.com/ErdosNumber.html), [Bacon number](https://simple.wikipedia.org/wiki/Bacon_number).

### Connectedness in Undirected Graphs

<div class="definition">
An <strong>undirected graph</strong> is called <em>connected</em> if there is a path between every pair of distinct
vertices of the graph.
<br>
An <strong>undirected graph</strong> that is <em>not connected</em> is called <em>disconnected</em>.
<br>
We say that we disconnect a graph when we remove vertices or edges, or both, to produce a
disconnected subgraph.
</div>

<figure>
  <img src="https://walkenho.github.io/images/graph-theory-and-networkX-part2-fig1.jpg" align="center">
  <figcaption>Connected vs Disconnected Graph (<a href="https://walkenho.github.io/graph-theory-and-networkX-part2/">walkenho</a>)</figcaption>
</figure>

<div clas="theorem">
There is a simple path between every pair of distinct vertices of a connected undirected graph.
</div>

<div class="definition">
A <strong>connected component</strong> of a graph G is a connected subgraph of G that is not a proper subgraph of another connected subgraph of G.
</div>
That is, a connected component of a graph G is a maximal connected subgraph of G.

A graph G that is not connected has two or more connected components that are disjoint and have G as their union.

<figure>
  <img src="https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/46457/versions/3/screenshot.jpg" width="75%">
  <figcaption>Connected Components (<a href="https://www.mathworks.com/matlabcentral/fileexchange/46457-splitting-a-network-into-connected-components">mathworks</a>)</figcaption>
</figure>

Read more: [Finding Connected Components using DFS](https://www.baeldung.com/cs/graph-connected-components#finding-connected-components)

### How Connected is a Graph?

<div class="definition">
<strong>Cut vertices</strong> or <strong>Articulation points</strong> is the vertices that if they and their incident edges are removed from a graph, a subgraph with more connected components is produced.<br>
An edge whose removal produces a graph with more connected components than in the original graph is called a <strong>Cut edge</strong> or <strong>Bridge</strong>.
</div>

<figure>
    <img src="https://static.javatpoint.com/tutorial/graph-theory/images/connectivity3.png">
    <figcaption>The vertices b, c, e is cut vertices and the edge (c, e) is cut edge <br> (<a href="https://www.javatpoint.com/graph-theory-connectivity">javatpoint</a>)</figcaption>
</figure>

#### Vertex Connectivity

When a complete graph $K_n$ removes a vertex and all incidents to it, the resulting graph is a complete graph $K_{n-1}$ which also a connected graph.

<figure>
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/kargest-subset-of-graph-vertices-with-edges-of-2-or-more-colors-3.png">
    <figcaption>Remove a vertex from a $K_6$ makes a $K_5$ graph (<a href="https://www.geeksforgeeks.org/largest-subset-graph-vertices-edges-2-colors/">geeksforgeeks</a>)</figcaption>
</figure>

Connected graph having no cut vertex is a **nonseparable graph** which is more connected than a graph with a cut vertex.

<div class="definition"><br>
    A subset $V'$ of the vertex set $V$ of $G=(V,E)$ is a <strong>vertex cut</strong>, or <strong>separating set</strong>, if $G-V'$ is disconnected.
    <br>
    <strong>Vertex connectivity</strong> of a noncomplete graph $G$, denoted by $\kappa(G)$, is the minimum number of vertices in a vertex cut.
</div>

<figure>
    <img src="https://i.stack.imgur.com/3zwRU.png">
    <figcaption>The Vertex connectivity is 1, because removing the vertex A makes 2 connected components <br> (<a href="https://math.stackexchange.com/questions/1951447/find-a-graph-that-has-a-high-minimum-degree-but-low-connectivity-and-edge-conne">stackexchange</a>)</figcaption>
</figure>

When $G$ is a complete graph, it has no vertex cuts. Because removing any vertex and its incident edges still leaves a complete graph.

So, it's impossible to define $\kappa(G)$ as the smallest number of vertices in a vertex cut when $G$ is complete. In this case, we define $\kappa(K_n) = n-1$ as the number of vertices needed to be removed to produce a graph with a single vertex.

In conclusion, for every graph G, $\kappa(G)$ is the minimum number of verties that can be removed from graph to either:
* Disconnect G or
* Prduce a graph with a single vertex

We have:
* $0 \le \kappa(G) \le n-1$ if G has n vertices
* $\kappa(G) = 0$ if and only if G is disconnected or $G = K_1$
* $\kappa(G) = 1$ if and only if G is complete

The larger k(G) is, the more connected G is:
* Disconnected graph or $K_1$ have $\kappa(G) = 0$
* Connected graph with cut vetices and $K_2$ have $\kappa(G) = 1$
* Graph without cut vertices and $K_3$ have $\kappa(G) = 2$
* so on ...

A graph is **k-connected** or **k-vertex-connected** if $\kappa(G) \ge k$:
* 1-connected graph if it is connected and not a graph containing a single vertex
* 2-connected is also called **biconnected** that is non seperable and has at least three vertices
* G is a k-connected graph, then G is a j-connected for all j with $0 \le j \le k$

#### Edge Connectivity

<div class="definition"><br>
    A set of edge E' is called an <strong>edge cut</strong> of G if the subgraph $G - E'$ is disconnected.
    <br>
    The <strong>edge connectivity</strong> of a graph G, denoted $\lambda(G)$, is the minimum number of edges in an edge cut of G.
</div>

We always define $\lambda(G)$ for any connected graphs with at least one vertex because it is always possible to disconnect a graph by removing all edges incident to one of its vertices.
* $\lambda(G) = 0$ if G is not connected.
* If G is a graph with $n$ vertices, then $0 \le \lambda(G) \le n - 1$
* $\lambda(G) = n - 1$ *if and only if* $G = K_n$
* $\lambda(G) \le n - 2$ *if and only if* G is **not** a complete graph.

<div class="theorem">
    When $G = (V,E)$ is a noncomplete connected graph with at least three vertices:
    $$\kappa(G) \le \lambda(G) \le \min_{v \in V} \deg(v) $$
</div>

Proof: [[1]](https://math.stackexchange.com/questions/1441288/inequality-relating-connectivity-edge-connectivity-and-minimum-degreewhitneys), [[2]](http://www.math.nagoya-u.ac.jp/~richard/teaching/s2020/Suzuki_Quang.pdf)

Read more:
* [Connectivity algorithm](http://www.cse.msu.edu/~esfahani/book_chapter/Graph_connectivity_chapter.pdf)
* [Edge and Vertex Connectivity](http://algorist.com/problems/Edge_and_Vertex_Connectivity.html)

### Connectedness in Directed Graphs

<div class="definition"><br>
    A directed graph is <strong>strongly connected</strong> if there is a path from a to b and from b to a whenever a and b are vertices in the graph. <br>
    A directed graph is <strong>weakly connected</strong> if there is a path between every two vertices in the underlying undirected graph.
</div>

A directed graph is weakly connected *if and only if* there is always a path between two vertices when the directions of the edges are disregarded. Clearly, any strongly connected directed graph is also weakly connected.

<figure>
    <img src="https://www.mssqltips.com/tipimages2/6746_graph-analytics-using-apache-spark-graphframe-api.025.png">
    <figcaption>Strong connected and Weakly connected graph (<a href="https://www.mssqltips.com/sqlservertip/6746/graph-analytics-apache-spark-graphframe-api/">mssqltips</a>)</figcaption>
</figure>

<div class="definition">
    The <strong>strongly connected components</strong> or <strong>strong components</strong> of $G$ are the maximal strongly connected subgraphs which is the subgraphs of a directed graph $G$ that are strongly connected but not contained in larger strongly connected subgraphs.
</div>

<figure>
    <img src="https://upload.wikimedia.org/wikipedia/commons/5/5c/Scc.png">
    <figcaption>
        Graph with strongly connected components marked (<a href="https://en.wikipedia.org/wiki/Strongly_connected_component">Wikipedia</a>)
    </figcaption>
</figure>

Read more:
* [Kosaraju’s algorithm](https://www.geeksforgeeks.org/strongly-connected-components/)
* [Tarjan's algorithm](https://www.youtube.com/watch?v=wUgWX0nc4NY)

### Paths and Isomorphism

<img src="https://rpruim.github.io/m252/S19/from-class/images/GraphFig9-4-6.png">

Both $G$ and $H$ have same three invariants: number of vertices, number of edges, and degrees of vertices—all agree for the two graphs. 

However, $H$ has a simple circuit of length three, namely, $v_1, v_2, v_6, v_1$, whereas $G$ has no simple circuit of length three, as can be determined by inspection (all simple circuits in $G$ have length at least four). Because the existence of a simple circuit of length three is an isomorphic invariant, $G$ and $H$ are not isomorphic.

<img src="https://rpruim.github.io/m252/S19/from-class/images/GraphFig9-4-7.png">

Both G and H have five vertices and six edges, both have two vertices of degree three and three vertices of degree two, and both have a simple circuit of length three, a simple circuit of length four, and a simple circuit of length five. Because all these isomorphic invariants agree, G and H may be isomorphic.

To confirm isomorphism, find the path go through all same degrees of 2 graphs. For example, the paths $u_1 \rightarrow u_4 \rightarrow u_3 \rightarrow u_2 \rightarrow u_5$ in $G$ and $v_3 \rightarrow v_2 \rightarrow v_1 \rightarrow v_5 \rightarrow v_4$ in $H$ both go through degrees: $3 \rightarrow 2 \rightarrow 3 \rightarrow 2 \rightarrow 2$. So, we can establish a mapping $f$ with $f(u_1) = v_3$, $f(u_4) = v_2$, $f(u_3) = v_1$, $f(u_2) = v_5$, and $f(u_5) = v_4$, this leads to that $G$ and $H$ are isomorphic.

### Counting Paths Between Vertices

<div class="definition">
    Let G be a graph with adjacency matrix $A$ with respect to the ordering $v_1, v_2, \dots, v_n$ of the vertices of the graph (with directed or undirected edges, with multiple edges and loops allowed).<br>
    The number of different paths of length $r$ from $v_i$ to $v_j$, where $r$ is a positive integer, equals the $(i, j)$th entry of $A^r$.
</div>

For example ([sfu](https://www.cs.sfu.ca/~ggbaker/zju/math/paths.html)):

The graph :

![](https://www.cs.sfu.ca/~ggbaker/zju/math/img/graph-path3.svg)

Adjacency Matrix:

$$M=\left[\begin{smallmatrix}1&1&0&0&1 \\ 1&0&0&0&0 \\ 0&1&0&1&0 \\ 0&1&1&0&0 \\ 1&0&0&0&1 \end{smallmatrix}\right]$$

The number of paths of length 2 between each pair of vertices:

$$M^2=\left[\begin{smallmatrix}3&1&0&0&2 \\ 1&1&0&0&1 \\ 1&1&1&0&0 \\ 1&1&0&1&0 \\ 2&1&0&0&2\end{smallmatrix}\right]$$

## Euler and Hamilton Paths

### Euler Paths and Circuits

### Hamilton Paths and Circuits


