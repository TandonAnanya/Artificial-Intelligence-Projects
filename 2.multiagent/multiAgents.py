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
        ghost_score = float(0)
        ghost_distance = [manhattanDistance(x.getPosition(), newPos) for x in newGhostStates]
        x = newGhostStates[-1]
        
        scores = []
        for i in ghost_distance:
            if (i>1):
                scores.append(0)                        
            elif not x.scaredTimer: 
                scores.append(-200)
            else:
                scores.append(1500)
        ghost_score += sum(scores)*(1.0)
            
        distanceToCapsule=[]
        for x in currentGameState.getCapsules():
          b=manhattanDistance(x,newPos)
          distanceToCapsule.append(b)         

        scores = [100 if not b else 10.0/b for _ in distanceToCapsule]        
        ghost_score +=  sum(scores)                 
        positionOfFood = [manhattanDistance(k,newPos) for k in (currentGameState.getFood()).asList()]        
        scores = [100 if not i else 1.0/(i**2) for i in positionOfFood]        
        ghost_score +=  sum(scores)         
        
        return ghost_score 

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
    def baseCondition(self, args):
        return True if any(args) else False

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
        numberOfAgents = gameState.getNumAgents()
        numberOfGhosts = numberOfAgents - 1

        def maxLevel(gameState,depth):            
            if self.baseCondition([gameState.isWin(), gameState.isLose(), depth + 1==self.depth]):
                return self.evaluationFunction(gameState)                    
            values = [minLevel(gameState.generateSuccessor(0,action),depth + 1,1) for action in gameState.getLegalActions(0)]
            return max(float("-inf"), max(values))

        def minLevel(gameState,depth, agentIndex):            
            if self.baseCondition([gameState.isWin(),gameState.isLose()]):
                return self.evaluationFunction(gameState)                        
            values = [maxLevel(gameState.generateSuccessor(agentIndex,action),depth) if agentIndex == numberOfGhosts else minLevel(gameState.generateSuccessor(agentIndex,action),depth,agentIndex+1) for action in gameState.getLegalActions(agentIndex)]
            return min(float('inf'), min(values))        

        def calculateAction(gameState):
            actionScore = [(action, minLevel(gameState.generateSuccessor(0,action),0,1)) for action in gameState.getLegalActions(0)]
            maxActionScore = max(actionScore, key=lambda x: (x[1]))
            return maxActionScore[0]
        
        return calculateAction(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def baseCondition(self, args):
        return True if any(args) else False

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numberOfAgents = gameState.getNumAgents()
        numberOfGhosts = numberOfAgents - 1

        def maxLevel(gameState,depth,alpha,beta):            
            if self.baseCondition([gameState.isWin(), gameState.isLose(), depth + 1==self.depth]):
                return self.evaluationFunction(gameState)            
            maxvalue = float("-inf")            
            for action in gameState.getLegalActions(0):                
                maxvalue = max(maxvalue,minLevel(gameState.generateSuccessor(0,action),depth + 1,1,alpha,beta))
                if maxvalue > beta:
                    return maxvalue
                alpha = max(maxvalue, alpha)
            return maxvalue
        
        def minLevel(gameState,depth,agentIndex,alpha,beta):
            if self.baseCondition([gameState.isWin(), gameState.isLose()]):
                return self.evaluationFunction(gameState)
            minvalue = float("inf")
            for action in gameState.getLegalActions(agentIndex):                
                if agentIndex == numberOfGhosts:
                    minvalue = min(minvalue,maxLevel(gameState.generateSuccessor(agentIndex,action),depth,alpha,beta))                    
                else:
                    minvalue = min(minvalue,minLevel(gameState.generateSuccessor(agentIndex,action),depth,agentIndex+1,alpha,beta))
                if minvalue < alpha:
                    return minvalue
                beta = min(beta,minvalue)                    
            return minvalue
                       
        actionScore = ['', float("-inf")]         
        pruningVariables = [float("-inf"), float("inf")]        
        for action in gameState.getLegalActions(0):
            score = minLevel(gameState.generateSuccessor(0,action),0,1,pruningVariables[0],pruningVariables[1])            
            if score > pruningVariables[1]:
                return actionScore[0]
            if score > actionScore[1]:
                actionScore[0] = action
                actionScore[1] = score
            pruningVariables[0] = max(pruningVariables[0],score)
        return actionScore[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def baseCondition(self, args):
        return True if any(args) else False


    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numberOfAgents = gameState.getNumAgents()
        numberOfGhosts = numberOfAgents - 1
        
        def maxLevel(gameState,depth):            
            if self.baseCondition([gameState.isWin(), gameState.isLose(), depth + 1==self.depth]):
                return self.evaluationFunction(gameState)            
            values = [expectLevel(gameState.generateSuccessor(0,action),depth + 1,1) for action in gameState.getLegalActions(0)]
            return max(values) if values else float("-inf")
        
        def expectLevel(gameState,depth, agentIndex):
            if self.baseCondition([gameState.isWin(), gameState.isLose()]):
                return self.evaluationFunction(gameState)                        
            values = [maxLevel(gameState.generateSuccessor(agentIndex,action),depth) if agentIndex == numberOfGhosts else expectLevel(gameState.generateSuccessor(agentIndex,action),depth,agentIndex+1) for action in gameState.getLegalActions(agentIndex)]
            return sum(values)/len(values) if values else 0
                    
        def calculateAction(gameState):
            actionScore = [(action, expectLevel(gameState.generateSuccessor(0,action),0,1)) for action in gameState.getLegalActions(0)]
            maxActionScore = max(actionScore, key=lambda x: (x[1]))
            return maxActionScore[0]
        
        return calculateAction(gameState)       