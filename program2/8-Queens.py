import random


class Board:
    def __init__(self, n):
        self.totalqueens = n
        self.totalPositon = n * n
        self.constraints = [0] * self.totalPositon
        self.queen = []

    def get_possible_moves(self):
        """
        获取当前状态下可能的移动策略
        """
        possibleMoves = []
        for move, numOfConstraints in enumerate(self.constraints):
            if numOfConstraints == 0:
                possibleMoves.append(move)
        return possibleMoves

    def make_move(self, move):
        """
        移动
        move: int 一维棋盘中的位置
        """
        self.queen.append(move)
        self.add_or_remove_constraints(move, add=True)

    def remove_move(self, move):
        """
        撤销移动
        move: int 一维棋盘中的位置
        """
        self.queen.remove(move)
        self.add_or_remove_constraints(move, add=False)

    def add_or_remove_constraints(self, move, add):
        """
        检查同一行、同一列、上对角线、下对角线是否有冲突，并执行或撤销移动
        """
        if add:
            callFunction = self.add_constraint
        else:
            callFunction = self.remove_constraint

        row = move // self.totalqueens
        col = move % self.totalqueens
        updiag = row + col
        lodiag = row - col

        for i in range(self.totalqueens):
            callFunction(self.get_pos(row, i))
            callFunction(self.get_pos(i, col))
            if updiag > -1:
                callFunction(self.get_pos(updiag, i))
                updiag -= 1
            if lodiag < self.totalqueens:
                callFunction(self.get_pos(lodiag, i))
                lodiag += 1

    def add_constraint(self, move):
        if not move == -1:
            self.constraints[move] += 1

    def remove_constraint(self, move):
        if not move == -1:
            self.constraints[move] -= 1

    def get_pos(self, row, col):
        """
        根据棋盘坐标获取其在一维列表中的位置，若非法则返回-1。
        例如：(1,3) = 1*8 + 3 = 11
        """
        pos = row * self.totalqueens + col
        if pos >= self.totalPositon or pos < 0:
            return -1
        else:
            return pos

    def print_board(self):
        """
        在终端打印棋盘
        """
        print('*' * 20)
        for i in range(self.totalqueens):
            row = ''
            for j in range(self.totalqueens):
                if self.get_pos(i, j) in self.queen:
                    row += 'Q'
                else:
                    row += '-'
                row += '  '
            print(row)

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
# 回溯法求解八皇后问题
def solveNQueens(b, j):
    """
    b: 当前棋盘
    j: 起始搜索所在行数
    return: True or False
    """
    valid_points = b.get_possible_moves()
    for point in valid_points:
        b.make_move(point)
        if dfs(b, point) == True:
            return True
        b.remove_move(point)
    return False

"""  valid_lines = [1] * 8
    valid_lines[j - 1] = 0
    last_move = 0
    
    while 1:
        valid_points = b.get_possible_moves()
        #find most constraint line
        #val_line, val_points_in_line = valuable_line(valid_points, valid_lines)
        
        if val_line == 0:
            b.remove_move(last_move)
        else:
            for point in val_points_in_line:
                b.make_move(point)

        #print(valid_points_in_line)
        #return False 
        
    #print(list) """



if __name__ == '__main__':

    b = Board(8)
    r = int(input("请你指定第一行皇后的摆放位置（0-7）："))
    while r < 0 or r > 7:
        r = int(input("输入无效。请输入第一位皇后的摆放位置（0-7）："))
    b.make_move(r)

    if solveNQueens(b, j=1):
        print('求解成功！')
        b.print_board()
    else:
        print('无解！')
