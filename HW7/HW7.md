### T2

1. 状态空间$S$与状态转移矩阵$P$如下所示
   $$
   S = \{(0, 0), (-1, 1), (1, -1), (-2, 2), (2, -2)\}\\
   \textbf{P} = 
    \left[
    \begin{matrix}
      r & q & p & 0 & 0 \\
      p & r & 0 & q & 0 \\
      q & 0 & r & 0 & p \\
      0 & 0 & 0 & 1 & 0 \\
      0 & 0 & 0 & 0 & 1 \\
     \end{matrix}
     \right]   \ \ \ 
   \textbf{P}^T = 
    \left[
    \begin{matrix}
      r & p & q & 0 & 0 \\
      q & r & 0 & 0 & 0 \\
      p & 0 & r & 0 & 0 \\
      0 & q & 0 & 1 & 0 \\
      0 & 0 & p & 0 & 1 \\
     \end{matrix}
     \right]
   $$
   

2. 
   $$
   p_2 = (\textbf{P}^T) ^2 \cdot p_0 = [2qr, q^2, r^2 + pq, 0, pr + p]\\
   p_1 = \textbf{P}^T \cdot p_0 = [q, 0, r, 0, p]\\
   $$
   所以，再赛两局之内可以结束比赛的概率为：$sum(p_2[3 : ]) = pr + p$

   刚好在第二局结束比赛的概率为：$sum(p2[3:]) - sum(p_1[3:]) = pr$

### T3

$$
\textbf{S} = \{1, 2, 3, 4, 5, 6, 7\}\\
\textbf{A} = \{up, down, left, right\} \\
\textbf{R} = [-1]\\
$$

$$
q_\pi (6, up) = E[G_t | S_t = 6, A_t = up]\\
q_\pi(6, up) = -1 + \sum p^a_{ss'}v_\pi(s') = -1 + v_\pi(3)\\

\begin{align}
v_\pi(3) &= \frac{1}{4}[(-1 + v_\pi(3)) + (-1 + 0) + (-1 + v_\pi(4)) + (-1 + v_\pi(6))]\\
v_\pi(4) &= \frac{1}{4} \cdot 4 \cdot (-1 + v_\pi(3))\\
v_\pi(6) &= \frac{1}{4} [(-1 + v_\pi(3)) * 2 + (-1 + v_\pi(6)) * 2]\\
\end{align}\\
\therefore v_\pi(3) = -7\\
\therefore q_\pi(6, up) = -8\\
由对称性，q_\pi(5, down) = q_\pi(3, up) = -1\\
$$

### T2

1. 
   $$
   S = \{A, B, C\}\\
   P = \left[
    \begin{matrix}
      0 & 0.5 & 0.5 \\
      0.5 & 0 & 0.5 \\
      0 & 0 & 1 \\
     \end{matrix}
     \right]\\
   \textbf{r} = [-1, -1, 0]^T\\
   \begin{align}
   \textbf{v} = (I - \gamma P)^{-1}\textbf{r} = [-1.33, -1.33, 0]^T\\
   \end{align}
   $$

2. 如果模型复杂，可以参考数值分析与算法中学到过的迭代法解方程（组）的方法进行数值求解，通过多次迭代，得到模型的近似（数值）解。

