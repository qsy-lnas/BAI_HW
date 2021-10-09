class TicTacToe:
    """
    井字棋类
    属性:
        self.gameboard (list): 棋盘（一维）
        self.turn (list): 轮到哪一方走棋 （X 或者 O）
        self.player (str): 玩家使用的字符（X 或者 O）
        self.computer (str): 电脑使用的字符（X 或者 O）
        self.turnNum(int): 当前总的轮次数
        self.lastindex(dic): holds the win and loss move for the computer if a win isnt possible
    参数:
        playerMark(int): 初始化时指定玩家使用的字符（0-X 或者 1-O）
    """

    def __init__(self, playerMark):
        self.gameboard = ['-', '-', '-', '-', '-', '-', '-', '-', '-']
        self.turn = ['X', 'O']
        self.player = self.turn[playerMark]
        self.computer = self.turn[(playerMark + 1) % 2]
        self.turnNum = 0
        self.lastindex = {}

    def print_board(self):
        '''
        输出棋盘
        '''
        gb = self.gameboard
        print(gb[0] + ' | ' + gb[1] + ' | ' + gb[2]
              + '\n---------\n'
              + gb[3] + ' | ' + gb[4] + ' | ' + gb[5]
              + '\n---------\n'
              + gb[6] + ' | ' + gb[7] + ' | ' + gb[8]
              + '\n')

    def get_turn(self, movenum):
        """
        决定下一个走棋的是人还是机器
        movenum: 当前的总轮次
        """
        return self.turn[movenum % 2]

    def game_complete(self):
        """
        检查胜出玩家并输出最终结果
        """
        winner = ''
        for i in range(3):
            if self.gameboard[3 * i] == self.gameboard[(3 * i) + 1] == self.gameboard[(3 * i) + 2]:
                winner = self.gameboard[3 * i]
                break
            elif self.gameboard[i] == self.gameboard[i + 3] == self.gameboard[i + 6]:
                winner = self.gameboard[i]
                break
        if self.gameboard[0] == self.gameboard[4] == self.gameboard[8] or self.gameboard[2] == self.gameboard[4] == \
                self.gameboard[6]:
            winner = self.gameboard[4]
        if winner == self.player:
            print("恭喜！你赢了！")

        elif winner == self.computer:
            print("啊哦……电脑赢了")
        else:
            print('平局！')

    def check_result(self, gameboard=None):
        """
        检查当前游戏是否已经结束

        """
        gb = gameboard
        if gb is None:
            gb = self.gameboard

        if '-' not in gb:
            return 0

        elif gb[0] == gb[1] == gb[2] and gb[0] != '-' or gb[3] == gb[4] == gb[5] and gb[3] != '-' or gb[6] == gb[7] == \
                gb[8] and gb[6] != '-':
            return 1

        elif gb[2] == gb[5] == gb[8] and gb[2] != '-' or gb[1] == gb[4] == gb[7] and gb[1] != '-' or gb[0] == gb[3] == \
                gb[6] and gb[0] != '-':
            return 1

        elif gb[2] == gb[4] == gb[6] and gb[2] != '-' or gb[0] == gb[4] == gb[8] and gb[0] != '-':
            return 1

        return -1

    def make_comp_move(self):
        """
        电脑走棋

        """
        move = None
        # 请在此处补全代码
        self.make_move(move)

    def make_move(self, move):
        """
        在指定位置走棋
        参数:
            move: 指定位置，为0-8的整数，分别棋盘从左至右、从上至下的位置。
        """
        if self.check_result(self.gameboard) >= 0:
            self.game_complete()
            return
        if 0 <= move <= 8 and '-' in self.gameboard:
            self.gameboard[move] = self.turn[self.turnNum % 2]
            self.turnNum += 1
        self.print_board()

    def make_player_move(self):
        """
        玩家走棋
        """
        move = None
        if '-' not in self.gameboard or self.check_result(self.gameboard) >= 0:
            return
        while move is None:
            print("=" * 35)
            temp = (int(input("玩家走棋，请输入落子位置（0-8）：")))
            if (0 <= temp <= 8) and self.gameboard[temp] == '-':
                move = temp
            else:
                print("此位置无效")

        self.make_move(move)


if __name__ == '__main__':
    play = True
    alternate = -1
    while play:
        alternate += 1
        game = TicTacToe(alternate % 2)
        game.print_board()
        while game.check_result() < 0:
            if alternate % 2 == 0:
                game.make_player_move()
                game.make_comp_move()

            else:
                game.make_comp_move()
                game.make_player_move()

        if input("是否重新开始？（请输入是或否）") != '是':
            play = False
