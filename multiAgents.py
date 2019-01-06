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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        if(successorGameState.isLose()):
            return(float("-inf"))
        elif(successorGameState.isWin()):
            return(float("inf"))
        if(len(successorGameState.getCapsules()) < len(currentGameState.getCapsules()) or len(newFood.asList()) < len(prevFood.asList())):
            return(float('inf'))
        if(successorGameState.getPacmanPosition() == currentGameState.getPacmanPosition()):
            return(-5)
        closest_ghost_dist = manhattanDistance(newPos, newGhostStates[0].getPosition())
        closest_ghost = newGhostStates[0]
        closest_food_dist = manhattanDistance(newPos, newFood.asList()[0])
        closest_food = newFood.asList()[0]
        for i in newGhostStates[1:]:
            if(manhattanDistance(newPos, i) < closest_ghost_dist):
                closest_ghost_dist = manhattanDistance(newPos, i)
                closest_ghost = i
        for i in newFood.asList()[1:]:
            if(manhattanDistance(newPos, i) < closest_food_dist):
                closest_food_dist = manhattanDistance(newPos, i)
                closest_food = i
        if(closest_ghost_dist <= 2):
            return(float("-inf"))
        successorGameState.data.score = closest_ghost_dist/closest_food_dist


        return successorGameState.getScore()


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
        self.pacmanIndex = 0

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
        curDepth = 0
        currentAgentIndex = 0
        val = self.value(gameState, currentAgentIndex, curDepth)
        return val[0]

    def value(self, gameState, currentAgentIndex, curDepth):
        if currentAgentIndex >= gameState.getNumAgents():
            currentAgentIndex = 0
            curDepth += 1

        if curDepth == self.depth:
            return self.evaluationFunction(gameState)

        if currentAgentIndex == self.pacmanIndex:
            return self.maxValue(gameState, currentAgentIndex, curDepth)
        else:
            return self.minValue(gameState, currentAgentIndex, curDepth)

    def minValue(self, gameState, currentAgentIndex, curDepth):
        v = ("unknown", float("inf"))

        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue

            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1]

            vNew = min(v[1], retVal)

            if vNew is not v[1]:
                v = (action, vNew)

        #print "Returning minValue: '%s' for agent %d" % (str(v), currentAgentIndex)
        return v

    def maxValue(self, gameState, currentAgentIndex, curDepth):
        v = ("unknown", -1*float("inf"))

        if not gameState.getLegalActions(currentAgentIndex):
            return self.evaluationFunction(gameState)

        for action in gameState.getLegalActions(currentAgentIndex):
            if action == "Stop":
                continue

            retVal = self.value(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex + 1, curDepth)
            if type(retVal) is tuple:
                retVal = retVal[1]

            vNew = max(v[1], retVal)

            if vNew is not v[1]:
                v = (action, vNew)

        #print "Returning maxValue: '%s' for agent %d" % (str(v), currentAgentIndex)
        return v



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        x, action = self.alphaBeta(0, gameState, 0, 0, float('-inf'), float('inf'))
        return action

    def alphaBeta(self, agent, state, depth, rDepth, alpha, beta):
        """
          Computes the minimax score using DFS
            agent  - the index of the current agent to represent
                     index 0 is always a maximizer, the rest are adversaries
            state  - the current game state to evaluate
            depth  - the current depth of the expansion
                     this depth only increments after all agents have been considered
            rDepth - the recursion depth, to format debug messages
            alpha  - pacmans best score on path to root
            beta   - ghosts best score on path to root
        """
        padding = '   ' * rDepth

        # if we are at our depth limit or there are no moves, we are at a leaf node
        # compute and return the score of this state
        if( (depth == self.depth) or (len(state.getLegalActions(agent)) == 0)):
          score = self.evaluationFunction(state)
          return (score,0)

        optimalScore = float('-inf') if agent == 0 else float('inf')
        optimalAction = Directions.STOP

        for action in state.getLegalActions(agent):
          successor = state.generateSuccessor(agent, action)
          nextAgent = (agent+1) % state.getNumAgents()
          nextDepth = depth+1 if nextAgent == 0 else depth
          score,x = self.alphaBeta(nextAgent, successor, nextDepth, rDepth+1, alpha, beta)
          newOptimalScore = max(score, optimalScore) if agent == 0 else min(score, optimalScore)
          optimalAction = optimalAction if newOptimalScore == optimalScore else action
          optimalScore = newOptimalScore
          if agent == 0:
            if score > beta:
              break
            alpha = max(alpha, score)
          else:
            if score < alpha:
              break
            beta = min(beta, score)

        return (optimalScore,optimalAction)


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
        x,action = self.expectimax(0, gameState, 0, 0)
        return action

    def expectimax(self, agent, state, depth, rDepth):
        """
          Computes the expectimax score using DFS
            agent  - the index of the current agent to represent
                     index 0 is always a maximizer, the rest are adversaries
            state  - the current game state to evaluate
            depth  - the current depth of the expansion
                     this depth only increments after all agents have been considered
            rDepth - the recursion depth, to format debug messages
            call this with 0,gameState, 0, 0 to kick things off
        """
        padding = '   ' * rDepth
        numActions = len(state.getLegalActions(agent))

        # if we are at our depth limit or there are no moves, we are at a leaf node
        # compute and return the score of this state
        if( (depth == self.depth) or (numActions == 0)):
          score = self.evaluationFunction(state)
          return (score,0)

        optimalScore = float('-inf') if agent == 0 else float('inf')
        optimalAction = Directions.STOP
        p = 1.0 / numActions
        expectValue = 0

        for action in state.getLegalActions(agent):
          successor = state.generateSuccessor(agent, action)
          nextAgent = (agent+1) % state.getNumAgents()
          nextDepth = depth+1 if nextAgent == 0 else depth
          score,x = self.expectimax(nextAgent, successor, nextDepth, rDepth+1)
          if agent == 0:
            newOptimalScore = max(score, optimalScore)
            optimalAction = optimalAction if newOptimalScore == optimalScore else action
            optimalScore = newOptimalScore
          else:
            expectValue += ( p * score )

        if agent == 0:
          return (optimalScore,optimalAction)
        else:
          return (expectValue, 0)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    distanceToFood = []
    distanceToNearestGhost = []
    distanceToCapsules = []
    score = 0

    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()
    numOfScaredGhosts = 0

    pacmanPos = list(currentGameState.getPacmanPosition())

    for ghostState in ghostStates:
        if ghostState.scaredTimer is 0:
            numOfScaredGhosts += 1
            distanceToNearestGhost.append(0)
            continue

        gCoord = ghostState.getPosition()
        x = abs(gCoord[0] - pacmanPos[0])
        y = abs(gCoord[1] - pacmanPos[1])
        if (x+y) == 0:
            distanceToNearestGhost.append(0)
        else:
            distanceToNearestGhost.append(-1.0/(x+y))

    for food in foodList:
        x = abs(food[0] - pacmanPos[0])
        y = abs(food[1] - pacmanPos[1])
        distanceToFood.append(-1*(x+y))

    if not distanceToFood:
        distanceToFood.append(0)

    return max(distanceToFood) + min(distanceToNearestGhost) + currentGameState.getScore() - 100*len(capsuleList) - 20*(len(ghostStates) - numOfScaredGhosts)
 

# Abbreviation
better = betterEvaluationFunction
