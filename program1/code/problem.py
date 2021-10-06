import numpy as np
from copy import deepcopy
from utils import *
import sys


class Node(object):  # Represents a node in a search tree
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        #self.status = status
        if parent:
            self.depth = parent.depth + 1

    def child_node(self, problem, action):
        next_state = problem.move(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.g(self.path_cost, self.state,
                                   action, next_state))
        return next_node

    def path(self):
        """
        Returns list of nodes from this node to the root node
        """
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __repr__(self):
        return "<Node {}(g={})>".format(self.state, self.path_cost)
        

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state


class Problem(object):
    def __init__(self, init_state=None, goal_state=None):
        self.init_state = Node(init_state)
        self.goal_state = Node(goal_state)

    def actions(self, state):
        """
        Given the current state, return valid actions.
        :param state:
        :return: valid actions
        """
        pass

    def move(self, state, action):
        pass

    def is_goal(self, state):
        pass

    def g(self, cost, from_state, action, to_state):
        return cost + 1

    def solution(self, goal):
        """
        Returns actions from this node to the root node
        """
        if goal.state is None:
            return None
        return [node.action for node in goal.path()[1:]]

    def expand(self, node):  # Returns a list of child nodes
        return [node.child_node(self, action) for action in self.actions(node.state)]
    

class GridsProblem(Problem):
    def __init__(self,
                 n,
                 init_state=[[11, 9, 4, 15], 
                             [1, 3, 0, 12], 
                             [7, 5, 8, 6], 
                             [13, 2, 10, 14]],
                 goal_state=[[1, 2, 3, 4], 
                             [5, 6, 7, 8], 
                             [9, 10, 11, 12], 
                             [13, 14, 15, 0]]):
        super().__init__(init_state, goal_state)
        self.n = n

    def is_valid(self, loc):
        if -1 < loc[0] < self.n and -1 < loc[1] < self.n:
            return True
        else:
            return False

    def actions(self, state):
        empty_row, empty_col = np.where(np.array(state) == 0)[0][0], np.where(np.array(state) == 0)[1][0]
        candidates = [[empty_row-1, empty_col], [empty_row+1, empty_col],
                      [empty_row, empty_col-1], [empty_row, empty_col+1]]
        valid_candidates = [item for item in candidates if self.is_valid(item)]
        return valid_candidates

    def move(self, state, action):
        """  """
        empty_row, empty_col = np.where(np.array(state) == 0)[0][0], np.where(np.array(state) == 0)[1][0]
        new_state = deepcopy(state)
        new_state[empty_row][empty_col] = state[action[0]][action[1]]
        new_state[action[0]][action[1]] = 0
        return new_state

    def is_goal(self, state):
        return state == self.goal_state.state

    def g(self, cost, from_state, action, to_state):
        return cost + 1

def Mdis(pstate, tstate):
    ret = 0
    for i in range(4):
        for j in range(4):
            if pstate[i][j] == 0:
                continue
            x = [i, j]
            #print(x)
            #print(pstate[i][j])
            tstate = np.array(tstate)
            #print(np.argwhere(tstate == pstate[i][j]).flatten())
            y = np.argwhere(tstate == pstate[i][j]).flatten()
            ret += np.sum(np.abs(x-y))
    #print(ret)
    return ret



def search_with_info(problem):
    #store node
    Bstack = PriorityQueue(problem.init_state)
    #store state
    list = []
    problem.init_state.path_cost = Mdis(\
        problem.init_state.state, problem.goal_state.state)
    #Bstack.push(problem.init_state)
    list.append(problem.init_state.state)
    gameover = False
    while (Bstack.empty() != True):
        node = Bstack.pop()
        for action in problem.actions(node.state):
            #if state undiscover
            if (list.count(problem.move(node.state, action)) == 0):
                new_state = problem.move(node.state, action)
                #path_cost += g()
                path_cost = node.depth + 1
                #path_cost += h()
                path_cost += Mdis(new_state, problem.goal_state.state)
                if problem.is_goal(new_state) == True:
                    print(problem.solution(new_node))
                    gameover = True
                    break
                new_node = Node(new_state, node, action, path_cost)
                list.append(new_state)
                Bstack.push(new_node)
                print("[depth: %d, distance: %d, pc: %d]" %(new_node.depth, (path_cost - new_node.depth) / 2, path_cost))
        if gameover == True:
            break      
    print("有信息搜索。")


def search_without_info(problem):
    #store node
    queue = Queue()
    #store state
    list = []
    #problem.init_state.status = "discovered"
    queue.enqueue(problem.init_state)
    list.append(problem.init_state.state)
    #print(problem.init_state)
    #print(queue.empty())
    number = 0
    gameover = False
    while(queue.empty() != True):
        #print("while")
        node = queue.dequeue()
        for action in problem.actions(node.state):
            #if state undiscover
            #print(problem.actions(node.state))            
            if(list.count(problem.move(node.state, action)) == 0):
                #print(list.count(problem.move(node.state, action)) == 0)
                #discover it and store in queue and dict
                number += 1
                new_state = problem.move(node.state, action)
                new_node = Node(new_state, node, action)
                list.append(new_state)
                queue.enqueue(new_node)
                #print(new_state)
                if number == 20922789888000:
                    print("False")
                    gameover = True 
                    break
                if problem.is_goal(new_state) == True:
                    print(problem.solution(new_node))
                    gameover = True
                    break
        if gameover == True:
            break;
    print("无信息搜索")



if __name__ == "__main__":
    problem = GridsProblem(4)
    type = "with_info"

    if type == "without_info":
        search_without_info(problem)
    else :
        search_with_info(problem)


    