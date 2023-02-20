# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        Ghosts = len(newGhostStates)                                                #number of ghosts
        score = successorGameState.getScore()                                       #Get each successor's starting score

        if newFood.asList():                                                        #Given that there's still food in the grid
            if newPos in newFood.asList():                                          #If there's food in the new Position
                score += 10                                                         #add 10 points to the score
            else:                                                                   #If not,
                foodDist = float('inf')                                             #set distance to closest food to infinity
                for food in newFood.asList():
                    newDist = manhattanDistance(newPos, food)
                    if newDist < foodDist:
                        foodDist = newDist                                          #find the manhattan distance to the closest food and
                score += 10 / foodDist                                              #add a score of 10 or less, depending on how close the food is from the new Position

        if Ghosts == 1:                                                             #If there's one ghost agent
            ghostDist = manhattanDistance(newPos, newGhostStates[0].getPosition())  #find the manhattan distance to it
            if ghostDist > 0:                                                       #(to avoid division by zero)
                score -= 25 / ghostDist                                             #subtract a score of 25 or less.
                """I decided on 25 so that when the distance to the ghost agent is 2.5 or less, the score subtraction outweights the addition (which is max 10)
                That way, pacman avoids dangerous positions that are close to ghost agents even if there's food there"""
        elif Ghosts >= 2:
            GhostDist = []
            for i in range(len(Ghosts)):
                GhostDist.append(manhattanDistance(newPos, newGhostStates[i].getPosition()))
                if GhostDist[i] < 5 and ghostDist[i] > 0:
                    score -= 25 / ghostDist[i]
            #Same for 2 or more Agents, with the difference that the subtraction only happens when one or both ghost agents are relatively close (manhattan dist < 5) to pacman

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(0)
        SuccessorStates = []
        for a in actions:                                               #For each legal pacman action
            SuccessorStates.append(gameState.generateSuccessor(0, a))   #Save the corresponding successor state in the successor states list
        values = []
        for state in SuccessorStates:                                   
            values.append(self.minimax(state, 0, 1))                    #Save the score for each successor state in the values list
        
        #print(max(values))
        HighScore = values.index(max(values))                           #Get the index of the highest score
        return actions[HighScore]                                       #return the highest scoring action

    def minimax(self, gameState, depth, agentIndex):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return (self.evaluationFunction(gameState))

        if agentIndex == 0:                                                 #pacman's move
            actions = gameState.getLegalActions(0)
            SuccessorStates = []
            for a in actions:
                SuccessorStates.append(gameState.generateSuccessor(0, a))
            values = []
            for state in SuccessorStates:
                values.append(self.minimax(state, depth, 1))                #run the minimax recursion with the same depth, but for agent 1 (first ghost)
                
            return max(values)
        
        elif agentIndex <= gameState.getNumAgents() - 1:                    #ghost's move
            actions = gameState.getLegalActions(agentIndex)
            SuccessorStates = []
            for a in actions:
                SuccessorStates.append(gameState.generateSuccessor(agentIndex, a))
            
            values = []
            if agentIndex < gameState.getNumAgents() - 1:                           #if there are still other ghosts agents that haven't moved
                for state in SuccessorStates:
                    values.append(self.minimax(state, depth, agentIndex + 1))       #run the minimax recursion for them (same depth)

            else:                                                                   #if this is the last ghost agent to move
                for state in SuccessorStates:
                    values.append(self.minimax(state, depth + 1, 0))                #move to the next depth -- pacman's turn to move
            
            return min(values)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        a = -float('inf')
        b = float('inf')
        v = -float('inf')

        for action in gameState.getLegalActions(0):             #pacman's move -- for each legal action
            succ = gameState.generateSuccessor(0, action)       #get the corresponding successor
            newVal = self.minVal(succ, 0, 1, a, b)              #call minVal function for that successor
            if newVal > v:
                v = newVal                                      #save the max value of all successors
                #no need to check if v > b since b = inf
                dec = action                                    #keep the action of the max value
            a = max(a, v)
        return dec

    def minVal(self, gameState, depth, agentIndex, a, b):                   #ghost's move
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return (self.evaluationFunction(gameState))
        
        v = float('inf')                                                    #Given alpha-beta algorithm
        for action in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, action)
            if agentIndex < gameState.getNumAgents() - 1:                   #if there are still other ghosts agents that haven't moved
                v = min(v, self.minVal(succ, depth, agentIndex + 1, a, b))  #run the minVal recursion for them (same depth)
            else:                                                           #if this is the last ghost agent to move
                v = min(v, self.maxVal(succ, depth + 1, 0, a, b))           #move to the next depth -- pacman's turn to move so we call maxVal

            if v < a:
                return v
            b = min(b, v)

        return v

    def maxVal(self, gameState, depth, agentIndex, a, b):                   #pacman's move
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return (self.evaluationFunction(gameState))

        v = -float('inf')                                                   #Given alpha-beta algorithm
        for action in gameState.getLegalActions(agentIndex):
            succ = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.minVal(succ, depth, 1, a, b))                   #run the minVal recursion with the same depth, but for agent 1 (first ghost)

            if v > b:                                                       
                return v
            a = max(a, v)

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(0)
        SuccessorStates = []
        for a in actions:                                               #For each legal pacman action
            SuccessorStates.append(gameState.generateSuccessor(0, a))   #Save the corresponding successor state in the successor states list
        values = []
        for state in SuccessorStates:
            values.append(self.expectimax(state, 0, 1))                 #Save the score for each successor state in the values list
        
        #print(max(values))
        HighScore = values.index(max(values))                           #Get the index of the highest score
        return actions[HighScore]                                       #return the highest scoring action

    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return (self.evaluationFunction(gameState))

        if agentIndex == 0:                                                 #pacman's move
            actions = gameState.getLegalActions(0)
            SuccessorStates = []
            for a in actions:
                SuccessorStates.append(gameState.generateSuccessor(0, a))
            values = []
            for state in SuccessorStates:
                values.append(self.expectimax(state, depth, 1))             #run the expectimax recursion with the same depth, but for agent 1 (first ghost)
                
            return max(values)
        
        elif agentIndex <= gameState.getNumAgents() - 1:                    #ghost's move
            actions = gameState.getLegalActions(agentIndex)
            SuccessorStates = []
            for a in actions:
                SuccessorStates.append(gameState.generateSuccessor(agentIndex, a))
            
            values = []
            if agentIndex < gameState.getNumAgents() - 1:                           #if there are still other ghosts agents that haven't moved
                for state in SuccessorStates:
                    values.append(self.expectimax(state, depth, agentIndex + 1))    #run the expectimax recursion for them (same depth)

            else:                                                                   #if this is the last ghost agent to move
                for state in SuccessorStates:
                    values.append(self.expectimax(state, depth + 1, 0))             #move to the next depth -- pacman's turn to move

            sum = 0
            for v in values:                                                        #Get the sum of all child nodes' values
                sum += v
            
            return sum / len(values)                                                #divide by the number of child nodes to get their average value

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <Similar to the evaluation function of Q1:
    I add points to the score if food is close to pacman and I subtract points from the score if the ghost(s) are close to it, based on their manhattan distance.>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
        
    score = currentGameState.getScore()                                     #get the current State's starting score
    numGhosts = len(GhostStates)                                            #number of ghosts

    if food.asList():                                                       #Given that there's still food in the grid
        foodDist = float('inf')                                             #set distance to closest food to infinity
        for f in food.asList():
            newDist = manhattanDistance(pos, f)
            if newDist < foodDist:
                foodDist = newDist                                          #find the manhattan distance to the closest food and
        score += 10 / foodDist                                              #add a score of 10 or less, depending on how close the food is from the new Position
    
    if numGhosts == 1:                                                          #If there's one ghost agent
        ghostDist = manhattanDistance(pos, GhostStates[0].getPosition())        #find the manhattan distance to it
        if ghostDist > 0:                                                       #(to avoid division by zero)
            score -= 7 / ghostDist                                              #subtract a score of 7 or less.
            
    elif numGhosts >= 2:
        GhostDist = []
        for i in range(len(numGhosts)):
            GhostDist.append(manhattanDistance(pos, GhostStates[i].getPosition()))
            if GhostDist[i] < 5 and ghostDist[i] > 0:
                score -= 7 / ghostDist[i]
    #Same for 2 or more Agents, with the difference that the subtraction only happens when one or both ghost agents are relatively close (manhattan dist < 5) to pacman

    return score

# Abbreviation
better = betterEvaluationFunction
