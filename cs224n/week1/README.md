
# Word Vectors

**WordNet** - Thesaurus containing lists of Synonym sets and hypernyms(”is a” relationships)

![Untitled](assest/Untitled.png)

### Issue with Traditional NLP:

1. Words are regarded as discrete symbols(localist representation). So, if vocab size is 500,000. Word hotel is represented by a 500,000 one-hot vector.
2. No Word Relationship or similarity. Two one-hot vector is just orthogonal

### Representing Words by their context in DL:

Words Meaning is given by the words that frequently appear close-by

Word Vectors/Work embedding → word represented by vectors in high dimensional vector space(like 300). Cant visualize but can project into 2d but lose lot of information.

![Untitled](assest/Untitled%201.png)

## Word2Vec

Framework for learning word vectors.

1. Large corpus of text
2. Create a fixed vocab from the large corpus(some preprocess steps)
3. Represent Each work in the Fixed Vocab by a vector
4. Iterate through each word at position ‘t’ in the text, which will be the center word ‘c’ and context words are words outside ‘o’. Use similarity of c and o to calculate prob of o given c and once we see the context word ‘o’ keep adjusting the word vectors to maximize the prob.

![Untitled](assest/Untitled%202.png)
### Word2vec: Objective Function

For each position t=1m…,T, predict context words within a window of fixed size m, geven center word $w_j$. The likelihood function is given by:

$$L(\theta) = \prod_{t=1}^T \prod_{j \neq 0, -m \leq j \leq m} P(w_{t+j} | w_t; \theta)$$
where θ represents all the variables to be optimized.

The objective function J(θ) is the (average) negative log-likelihood:

$$J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^T \sum_{j \neq 0, -m \leq j \leq m} \log P(w_{t+j} | w_t; \theta)$$

The objective function is sometimes called a cost or loss function.
$\text{Minimizing objective function} \Leftrightarrow \text{Maximizing predictive accuracy}$

minimize objective function $J(\theta)$

$$J(\theta)=\frac{-1}{T} \sum_{t=1}^T \sum_{\substack{j \leq j \leq m \\ j \neq 0}} \log P\left(\omega_{t+j} \mid \omega_t ; \theta\right)$$

To calculate $P\left(w_{t+j} \mid w_t ; \theta\right)$, we use two vectors per word "w":

$v_w$ when w is a center word

$u_w$ when w is a context word.

Then for context ward "o", and center word "c"

$$P(o\mid c)=\frac{\exp \left(u_0^{\top} v_c\right)}{\sum_{w \in v} \exp \left(u^{\top}_w v_c\right)}$$
lager dot product = larger Probability

Dot product between o and c vectors, compares similarity

$$\theta=\left[\begin{array}{c}
\text { Vaardvert } \\
\vdots \\
\text { veewer } \\
\text { Ueaduak } \\
\vdots \\
\text { Uzebra }
\end{array}\right] \in R^{2 d v}$$

Compute Gradient w.rt $v_c$ and $u_o$

$$\begin{aligned}
& \frac{\partial}{\partial v_c} p(0 \mid c)=u_0-\sum_{x=1}^n \frac{\exp \left(u_x^T v_c\right)}{\sum_1^n \exp \left(u_w^T v_c\right)} u_w \\
& =u_0-\sum_{x=1}^n p(n \mid c) u_y=u_0-E\left[u_0\right] \\
& \text { observed - expected } \\
&
\end{aligned}$$


![Untitled](assest/image.png)
![Alt text](assest/image2.png)

Word2vec maximizes objective function by putting similar words nearby in space

So for a window ‘m’ in SGD, we only have at most 2m+1 words, making our gradient vector very sparse(so not efficient).

$$\nabla_{\theta} J_t(\theta)=\left[\begin{array}{c}
\text { 0 } \\
\vdots \\
\nabla_{v_{like}} \\
\vdots \\
\text { 0 } \\
\nabla_{u_{I}}  \\
\vdots \\
\nabla_{u_{learning}} \\
\vdots \\
\text { 0 } \\
\end{array}\right] \in R^{2 d v}$$

Words Vectors are row vector in DL frameworks.
