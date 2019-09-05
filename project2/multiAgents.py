# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    # print 'successorGameState: ', successorGameState
    newPos = successorGameState.getPacmanPosition()
    # print 'newPos: ', newPos
    newFood = successorGameState.getFood()
    # print 'newFood', newFood
    newGhostStates = successorGameState.getGhostStates()
    # print 'newGhostStates', newGhostStates
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # print 'newScaredTimes', newScaredTimes

    "*** YOUR CODE HERE ***"
    newGhostPosition = successorGameState.getGhostPositions()

    foodScore = 1
    if len(newFood.asList()) > 0:
      minDistance = float("inf")
      for foodPos in newFood.asList():
        minDistance = min(minDistance, util.manhattanDistance(newPos, foodPos))
      
      foodScore = 1 / float(minDistance)

    ghostScore = 0
    closestGhost = float("inf")
    for ghostPos in newGhostPosition:
      closestGhost = min(closestGhost, util.manhattanDistance(newPos, ghostPos))
    
    if closestGhost < 2: 
      ghostScore = -1000

    return successorGameState.getScore() - currentGameState.getScore() + foodScore + ghostScore

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def maximize(gameState, depth, agentIndex):
      if depth > self.depth or gameState.isLose() or gameState.isWin():
        return (self.evaluationFunction(gameState), None)
      actions = gameState.getLegalActions(agentIndex)
      maxVal = float("-inf")
      bestAction = Directions.STOP
      for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        score = minimize(successorState, depth + 1, agentIndex + 1)[0]
        if score > maxVal:
          maxVal = score
          bestAction = action
      return (maxVal, bestAction)

    def minimize(gameState, depth, agentIndex):
      if depth > self.depth or gameState.isLose() or gameState.isWin():
        return (self.evaluationFunction(gameState), None)
      actions = gameState.getLegalActions(agentIndex)
      minVal = float("inf")
      bestAction = Directions.STOP
      for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        score = 0
        if agentIndex < gameState.getNumAgents() - 1:
          score = minimize(successorState, depth, agentIndex + 1)[0]
        else:
          score = maximize(successorState, depth + 1, 0)[0]
        if score < minVal:
          minVal = score
          bestAction = action
      return (minVal, bestAction)

    return maximize(gameState, 1, 0)[1]
    util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def maximize(gameState, depth, agentIndex, alpha, beta):
      if depth > self.depth or gameState.isLose() or gameState.isWin():
        return (self.evaluationFunction(gameState), None)
      actions = gameState.getLegalActions(agentIndex)
      maxVal = float("-inf")
      bestAction = Directions.STOP
      for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        score = minimize(successorState, depth + 1, agentIndex + 1, alpha, beta)[0]
        if score > maxVal:
          maxVal = score
          bestAction = action
        if maxVal > beta:
          return (maxVal, action)
        alpha = max(alpha, maxVal)
      return (maxVal, bestAction)

    def minimize(gameState, depth, agentIndex, alpha, beta):
      if depth > self.depth or gameState.isLose() or gameState.isWin():
        return (self.evaluationFunction(gameState), None)
      actions = gameState.getLegalActions(agentIndex)
      minVal = float("inf")
      bestAction = Directions.STOP
      for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        score = 0
        if agentIndex < gameState.getNumAgents() - 1:
          score = minimize(successorState, depth, agentIndex + 1, alpha, beta)[0]
        else:
          score = maximize(successorState, depth + 1, 0, alpha, beta)[0]
        if score < minVal:
          minVal = score
          bestAction = action
        if minVal < alpha:
          return (minVal, action)
        beta = min(beta, minVal)
      return (minVal, bestAction)

    return maximize(gameState, 1, 0, float("-inf"), float("inf"))[1]
    util.raiseNotDefined()

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
    def value(gameState, depth, agentIndex):
      # if node is a terminal node
      if depth > self.depth or gameState.isLose() or gameState.isWin():
        return (self.evaluationFunction(gameState), None)
      # if node isn't a terminal node
      else:
        # if node is a max node 
        if agentIndex == 0:
          return maxValue(gameState, depth, agentIndex)
        else:
        # if node is a exp node
          return expValue(gameState, depth, agentIndex)
    
    def maxValue(gameState, depth, agentIndex):
      maxVal = float("-Inf")
      bestAction = Directions.STOP
      actions = gameState.getLegalActions(agentIndex)
      for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        # find all successors' values and return the max one
        # change agentIndex from 0 to 1 --> from pacman to ghost 
        score = value(successorState, depth + 1, agentIndex + 1)[0]
        if score > maxVal:
          maxVal = score
          bestAction = action
      return (maxVal, bestAction)
    
    def expValue(gameState, depth, agentIndex):
      # num of ghosts
      numGhosts = gameState.getNumAgents() - 1
      actions = gameState.getLegalActions(agentIndex)
      # num of actions
      numActions = len(actions)
      # weights = 1 / num of actions here --> probability
      weights = 1 / float(numActions)
      expectation = 0
      bestAction = Directions.STOP
      for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        if agentIndex == numGhosts: # all ghosts were found
          expectation += (value(successorState, depth + 1, 0)[0] * weights)
        else: # some ghosts left
          # stay on the same depth and search diff ghosts
          expectation += (value(successorState, depth, agentIndex + 1)[0] * weights)

      return (expectation, bestAction)
    
    return value(gameState, 1, 0)[1]
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    1. initialize score.
    2. calculate ghost distance.
    3. calculate food distance.
    4. finalize the score.
  """
  "*** YOUR CODE HERE ***"
  # current score
  currentScore = currentGameState.getScore()
  currentPosition = currentGameState.getPacmanPosition()
  currentFood = currentGameState.getFood()
  currentFoodList = currentFood.asList()
  # food distance between current position and current food
  minFoodDist = float("Inf")
  if len(currentFoodList) > 0:
    foodDist = []
    for food in currentFoodList:
      foodDist.append(manhattanDistance(currentPosition, food))
    # min food distance
    minFoodDist = min(foodDist)
    
  foodWeight = 8.0
  currentScore += (1.0 / minFoodDist) * foodWeight
  
  # ghost states
  ghostScore = 0
  ghostWeight = 8.0
  scaredGhostWeight = 80.0
  ghostStates = currentGameState.getGhostStates()
  ghostDist = []
  for ghostState in ghostStates:
    ghostDist.append(manhattanDistance(currentPosition, ghostState.getPosition()))
  minGhostDist = min(ghostDist)
  if minGhostDist > 0:
    # if ghost is scared: go to ghost.
    if ghostState.scaredTimer > 0:  
      ghostScore += scaredGhostWeight / minGhostDist
    # ghost is not scared: excape
    else:  
      ghostScore -= ghostWeight / minGhostDist
  currentScore += ghostScore

  return currentScore
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"


    util.raiseNotDefined()

