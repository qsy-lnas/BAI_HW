> 本部分知识点我觉得不易理解，故完成了全部四道题，如助教能拨冗批阅全部四道题，我能够对照更好的掌握知识点与复习我将不胜感激！

### T1

1. $$
   \begin{align}
   &\quad \ \neg P \Rightarrow\neg(P \Rightarrow Q)\\
   &\equiv (P \vee \neg (\neg P \vee Q))\\
   &\equiv P \vee (P \wedge Q)\\
   &\equiv P
   \end{align}
   $$

2. $$
   \begin{align}
   &\quad \ (\neg P \vee \neg Q) \Rightarrow (P \iff \neg Q)\\
   &\equiv \neg(\neg P \vee \neg Q) \vee ((P \Rightarrow \neg Q) \wedge (\neg Q \Rightarrow P))\\
   &\equiv (P \wedge Q) \vee ((\neg P \vee \neg Q) \wedge (Q \vee P))\\
   &\equiv (P \wedge Q) \vee (\neg P \wedge Q) \vee (P \wedge \neg Q)\\
   &\equiv \neg (\neg P \wedge \neg Q)\\
   &\equiv P \vee Q
   \end{align}
   $$

3. $$
   \begin{align}
   &\quad \ (\neg P \Rightarrow \neg Q) \Rightarrow (P \Rightarrow Q)\\
   &\equiv \neg(P \vee \neg Q) \vee (\neg P \vee Q)\\
   &\equiv (\neg P \wedge Q) \vee (\neg P \vee Q)\\
   &\equiv (\neg P \vee Q \vee \neg P) \wedge (\neg P \vee Q \vee Q)\\
   &\equiv \neg P \vee Q
   \end{align}
   $$

4. $$
   \begin{align}
   &\quad \ (P \wedge \neg Q \wedge S) \vee (\neg P \wedge Q \wedge R)\\
   &\equiv (P \vee Q) \wedge (P \vee R) \wedge (\neg Q \vee \neg P) \wedge (\neg Q \vee R) \wedge (S \vee \neg P) \wedge (S \vee Q) \wedge (S \vee R)
   \end{align}
   $$

   

### T3

“今天上人智课” = $\alpha$，“在澜园吃午饭” = $\beta$， 

“在一教上课” $ = x$， “在12点后吃午饭”$ = y$， “在清芬园吃午饭” = $z$，“清芬园人多” = $w$

知识库$KB$：$(\alpha \Rightarrow x) \wedge (\alpha \Rightarrow y) \wedge (x  \Rightarrow (\beta \vee z)) \wedge (y \Rightarrow w) \wedge (w \Rightarrow \neg z)$

试证明：$\alpha \Rightarrow \beta$
$$
\begin{align*}
& \alpha \Rightarrow x  & 前提引入\\
& \alpha & 前提引入\\
& x & 假言推理\\
& & \\
& \alpha \Rightarrow y  & 前提引入\\
& \alpha & 前提引入\\
& y & 假言推理\\
& & \\
& x  \Rightarrow (\beta \vee z) & 前提引入\\
& x & 前提引入\\
& \beta \vee z & 假言推理\\
& & \\
& y \Rightarrow w  & 前提引入\\
& y & 前提引入\\
& w & 假言推理\\
& & \\
& w \Rightarrow \neg z  & 前提引入\\
& w & 前提引入\\
& \neg z & 假言推理\\
& & \\
& \beta \vee z  & 前提引入\\
& \neg z & 前提引入\\
& \beta & 假言推理\\
& & \\
&\therefore \alpha \Rightarrow \beta & 证毕\\
\end{align*}
$$

### T2

1. $$
   \begin{align}
   A \Rightarrow B & \equiv \neg A \vee B\\
   \therefore (A \Rightarrow B) \wedge \neg B & \equiv (\neg A \vee B) \wedge \neg B\\
   & \equiv \neg A \vee (B \wedge \neg B)\\
   & \equiv \neg A\\
   \therefore ((A \Rightarrow B) \wedge \neg B & \Rightarrow \neg A) = True\\
   \end{align}
   $$

2. $$
   \begin{align}
   ((A \iff B) \wedge (B \iff C)) & \equiv ((A \Rightarrow B) \wedge (B \Rightarrow A) \wedge (B \Rightarrow C) \wedge (C \Rightarrow B))\\
   & \equiv (A \Rightarrow C) \wedge(C \Rightarrow A)\\
   & \equiv A \iff C\\
   \therefore (((A \iff B) \wedge (B \iff C)) &\Rightarrow  (A \iff C)) = True
   \end{align}
   $$

3. $$
   \begin{align}
   &A \Rightarrow B\ and\ B \Rightarrow C:\\
   &\qquad if\ A = 0:  \\
   &\qquad \qquad A \Rightarrow C\\
   &\qquad elif\ A = 1:\\
   &\qquad \qquad \because A \Rightarrow B \therefore B = 1\\
   &\qquad \qquad \because B \Rightarrow C \therefore C = 1\\
   &\qquad \qquad \therefore A \Rightarrow C\\
   &\therefore (((A \Rightarrow B) \wedge (B \Rightarrow C)) \Rightarrow (A \Rightarrow C) )= True
   \end{align}
   $$

4. $$
   \begin{align}
   & \qquad((A \Rightarrow B) \wedge (C \Rightarrow D) \wedge (\neg B \vee \neg D)) \Rightarrow (\neg A \vee \neg C)\\
   & \ \ \ \equiv ((A \Rightarrow B) \wedge (C \Rightarrow D) \wedge (B \Rightarrow \neg D)) \Rightarrow (A \Rightarrow \neg C)\\
   & \ \ \ \equiv ((A \Rightarrow \neg D) \wedge (C \Rightarrow D)) \Rightarrow (A \Rightarrow \neg C)\\
   & \ \ \ \equiv ((\neg A \vee \neg D) \wedge (\neg C \vee D)) \Rightarrow (\neg A \vee \neg C)\\
   & \ \ \ \equiv ((\neg A \wedge \neg C) \vee (\neg A \wedge D) \vee (\neg C \wedge \neg D) \vee False) \Rightarrow (\neg A \vee \neg C)\\
   & \ \ \ \equiv ((\neg C \wedge \neg D) \vee (\neg A \wedge D)) \Rightarrow (\neg A \vee \neg C)\\
   & \ \ \ \equiv True
   \end{align}
   $$



### T4

$$
\begin{align}
&\quad \ KB \wedge \neg \alpha\\
&\quad \ (A \Rightarrow C) \vee (B \Rightarrow C) \wedge \neg(A \vee B \Rightarrow C)\\
&\equiv ((\neg A \vee C) \vee (\neg B \vee C)) \wedge \neg ((\neg A \wedge \neg B) \vee C)\\
&\equiv ((\neg A \vee C) \vee (\neg B \vee C)) \wedge ((A \vee B) \wedge \neg C)\\
&\equiv (\neg A \vee C \vee \neg B) \wedge ((A \wedge \neg C) \vee (B \wedge \neg C))\\
&\equiv \neg B \vee (B \wedge \neg C)\\
&\equiv \neg B \vee \neg C\\
&\neq False\\
&\therefore KB \Rightarrow \alpha 不成立
\end{align}
$$

