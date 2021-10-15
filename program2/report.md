<h2  align = "center" >人工智能基础第二次编程<br> 实验报告 </h2>

<h6 align = "center">自96 曲世远 2019011455</h6>

### 1.作业要求

我本次作业选择的是完成八皇后问题，本题我采用的是回溯搜索的算法，由于问题复杂度较低，可以不采用较为复杂的回溯算法，只采用最基础的类似于添加了约束条件的DFS算法即可实现目标。

### 2.算法实现

```python
def dfs(b, point):
    valid_points = b.get_possible_moves()
    if len(valid_points) == 0:
        if len(b.queen) == 8:
            return True
        return False
    else:
        for point in valid_points:
            b.make_move(point)
            if dfs(b, point) == True:
                return True
            b.remove_move(point)
```

以上函数为我实现本题的主要算法，主要思路就是在递归到全部点时判断是否满足约束条件，如果不满足约束条件则返回继续搜索。

在本题最开始的尝试中，我也试图使用最少剩余项和最多约束项的方式进行回溯搜索，不过进行了复杂度分析之后，我觉得对于本题来说，上述的方式的代价与收益差别不大，于是还是采用了最基础的DFS改进的方式完成，下附最少约束项的确定方式：

```python
def valuable_line(points, valid_lines):
    #count the points in each line
    lines = [0] * 8
    #return line
    ret = 0
    #return points
    valid_points = []
    value = 8
    value_lines = [0] * 8
    for point in points:
        value_lines[point // 8] = 1
        lines[point // 8] += 1
    i = -1
    for line in lines:
        i += 1
        if valid_lines[i] * value_lines[i] == 0:
            continue
        if line < value:
            value = line
            ret = i + 1
    for point in points:
        if point // 8 == ret - 1:
            valid_points.append(point)
    return ret, valid_points
```

