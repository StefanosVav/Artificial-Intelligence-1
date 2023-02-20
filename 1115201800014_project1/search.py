# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    "*** YOUR CODE HERE ***"
    """

    if problem.isGoalState(problem.getStartState()):
        print ("Pacman is already at its goal state")
        return None
    
    explored = set()                            #set to check explored nodes

    fringe = util.Stack()                       #using the stack data structure from util.py
    fringe.push((problem.getStartState(), []))
    """
    My fringe is a stack of lists with 2 items: CurrState tuple and Path until current node (Assigned with getSuccessors function)
    The path variable is a list of directions to reach each node (updated every time we push a new item to the fringe)
    ex. For the starting state, the path until the starting node is an empty list.
    """

    while fringe.isEmpty() == False:
        CurrState, path = fringe.pop()          #'last-in' item in the fringe gets popped (assigns current state, path until current node)
        
        if CurrState not in explored:
            explored.add(CurrState)

            if problem.isGoalState(CurrState):
                return path

            for succ in problem.getSuccessors(CurrState):
                fringe.push((succ[0], path + [succ[1]]))    #each successor is a list [(num,num), direction, cost]. We want to assign the state tuple (num,num) as CurrState and add the direction to the path (until current node)

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    if problem.isGoalState(problem.getStartState()):
        print ("Pacman is already at its goal state")
        return None
    
    explored = set()                            #set to check explored nodes

    fringe = util.Queue()                       #using the queue data structure from util.py
    fringe.push((problem.getStartState(), []))
    """
    My fringe is a Queue of lists with 2 items: CurrState tuple and Path until current node (Assigned with getSuccessors function)
    The path variable is a list of directions to reach each node (updated every time we push a new item to the fringe)
    ex. For the starting state, the path until the starting node is an empty list.
    """

    while fringe.isEmpty() == False:
        CurrState, path = fringe.pop()          #'first-in' item in the fringe gets popped (assigns current state, path until current node)

        if CurrState not in explored:
            explored.add(CurrState)

            if problem.isGoalState(CurrState):
                return path

            for succ in problem.getSuccessors(CurrState):
                fringe.push((succ[0], path + [succ[1]]))    #each successor is a list [state, direction, cost]. We want to assign the state tuple as CurrState and add the direction to the path (until current node)

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    if problem.isGoalState(problem.getStartState()):
        print ("Pacman is already at its goal state")
        return None
    
    explored = set()                            #set to check explored nodes

    fringe = util.PriorityQueue()                       #using the priority queue data structure from util.py
    fringe.push((problem.getStartState(), [], 0), 0) 
    """
    For this function, my fringe is a priority queue that has items with: VALUE: state, path, totalcost / PRIORITY: totalcost
    The reason behind having totalcost appearing twice is that I had problems assigning the priority to my variable TotalCost,
    and thus totalcost is both part of an item's value and the item's priority.
    The path variable is a list of directions to reach each node (updated every time we push a new item to the fringe)
    ex. For the starting state, the path until the starting node is an empty list.
    """

    while fringe.isEmpty() == False:
        CurrState, path, TotalCost = fringe.pop()          #'highest priority/least total cost' item in the fringe gets popped (assigns current state, path until current node, totalcost)

        if CurrState not in explored:
            explored.add(CurrState)

            if problem.isGoalState(CurrState):
                return path

            for succ in problem.getSuccessors(CurrState):
                fringe.update((succ[0], path + [succ[1]], TotalCost + succ[2]), TotalCost + succ[2])    
                #each successor is a list [(num,num), direction, cost]. We want to assign the state tuple (num,num) as CurrState, add the direction to the path (until current node) and add the cost to the totalcost(both for priority and for the item)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    if problem.isGoalState(problem.getStartState()):
        print ("Pacman is already at its goal state")
        return None
    
    explored = set()                            #set to check explored nodes

    fringe = util.PriorityQueue()                       #using the priority queue data structure from util.py
    fringe.push((problem.getStartState(), [], 0), 0) 
    """
    For this function, my fringe is a priority queue that has items with: VALUE: state, path, totalcost / PRIORITY: combined cost and heuristic
    In the A* implementation we choose the path with the smallest combined cost and heuristic - that's the queue's priority.
    The path variable is a list of directions to reach each node (updated every time we push a new item to the fringe)
    ex. For the starting state, the path until the starting node is an empty list.
    """

    while fringe.isEmpty() == False:
        CurrState, path, TotalCost = fringe.pop()          #'highest priority/least total combined c+h' item in the fringe gets popped (assigns current state, path until current node, totalcost)

        if CurrState not in explored:
            explored.add(CurrState)

            if problem.isGoalState(CurrState):
                return path

            for succ in problem.getSuccessors(CurrState):
                CnH = (succ[2] + TotalCost) + heuristic(succ[0], problem)         #Combined (TotalCost) + Heuristic of the current state
                fringe.update((succ[0], path + [succ[1]], TotalCost + succ[2]), CnH)    
                #each successor is a list [(num,num), direction, cost]. We want to assign the state tuple (num,num) as CurrState, add the direction to the path (until current node), add the cost to the totalcost and assign the Cost+Heuristic to the priority

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
