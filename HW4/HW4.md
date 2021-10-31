### T2

1. $$
   \begin{align}
   &\quad \ \neg \ \exist x \ [P(x) \wedge Q(x)]\\
   &\equiv \forall x \ \neg[P(x) \wedge Q(x)]\\
   &\equiv \forall x \ [\neg P(x) \vee \neg Q(x)]\\
   &\equiv \forall x \ [P(x) \Rightarrow \neg Q(x)]\\
   &\therefore \neg \ \exist x \ [P(x) \wedge Q(x)] \ \iff \forall x\  [P(x) \Rightarrow \neg Q(x)]
   \end{align}
   $$

2. $$
   \begin{align}
   &\quad \ \neg \ \forall x \ [P(x) \Rightarrow Q(x)]\\
   &\equiv \exist x \ \neg[\neg P(x) \vee Q(x)]\\
   &\equiv \exist x \ [P(x) \wedge \neg Q(x)]\\
   &\therefore \neg \ \forall x \ [P(x) \Rightarrow Q(x)] \ \iff \exist x \ [P(x) \wedge \neg Q(x)]
   \end{align}
   $$

3. $$
   \neg \ \exist x \ [PN(x) <  NN(x)]\\
   \forall x\ [PN(x) > \ NN(x)]\\
   PN(x): positive\ number;\ NN(x):\ negative\ number\\
   $$

   $$
   \begin{align}
   &\quad \ \neg \ \exist x \ [PN(x) < NN(x)]\\
   &\equiv \forall x\ \neg [PN(x) < NN(x)]			\\
   &\equiv \forall x \ [PN(x) > NN(x)]\\
   &\therefore \neg \ \exist x \ [PN(x) < NN(x)] \ \iff \forall x \ [PN(x) > NN(x)]
   \end{align}
   $$

4. $$
   \neg \forall(x, y)\ [x == y \Rightarrow Diagonal(x, y)\\
   \exist (x, y)\ [x == y\ \wedge\ \neg Diagnal(x, y)] \\
   Diagnal(x, y): x, y\ is\ diagnal\\
   $$

   $$
   \begin{align}
   &\quad \ \neg \forall(x, y)\ [x == y \Rightarrow Diagonal(x, y)\\
   &\equiv \exist(x, y)\ \neg[(x != y) \vee Diagnal(x, y)]\\
   &\equiv \exist(x, y)\ [(x == y) \wedge \neg Diagnal(x, y)]\\
   &\therefore \neg \forall(x, y)\ [x == y \Rightarrow Diagonal(x, y) \iff \exist(x, y)\ [(x == y) \wedge \neg Diagnal(x, y)]
   \end{align}
   $$

   

### T4

前提：
$$
& \forall x\ [N(x) \Rightarrow (I(x)\wedge GZ(x))]\\
& \forall x\ [I(x) \Rightarrow (O(x) \vee E(x))]\\
& \forall x\ [S(E(x)) \Rightarrow I(x)]\\
$$
目标：
$$
\forall x\ [N(x) \Rightarrow (O(x) \vee S^{-1}(I(x)))]\\
$$
证明：

原子语句：
$$
\begin{align}
&\neg N(x) \vee I(x) & 1\\
&\neg N(x) \vee GZ(x)& 2\\
&\neg I(x) \vee O(x) \vee E(x) & 3\\
&\neg S(E(x)) \vee I(x) & 4\\
&\neg S^{-1}(I(x)) & 5\\
&\neg O(x) & 6\\
&N(x) & 7\\
&[S^{-1}(I(x)) \iff E(x)]
\end{align}
$$
归结：
$$
\begin{align}
1, 7: & I(x)\\
3: & E(x) \vee O(x)\\
6: & E(x)\\
5: & 空语句\\
原式得证
\end{align}
$$


### T1

1. $$
   \begin{align}
   &\quad \ \exist x\ \{P(x) \wedge \forall y\ [\neg Q(y) \vee R(x, y)]\}\\
   &\equiv \exist x\ \forall y \ \{P(x) \wedge  [\neg Q(y) \vee R(x, y)]\}
   \end{align}
   $$

2. $$
   
   $$

   
