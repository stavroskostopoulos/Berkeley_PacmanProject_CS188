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
        #print(action)
        #print('\nsuccessor:\n', successorGameState,'\n action:', action, '\ncurrent:\n', currentGameState)
        newPos = successorGameState.getPacmanPosition()
        #print(newPos)
        newFood = successorGameState.getFood()
        #print(newFood)
        #print(newFood.asList())
        newGhostStates = successorGameState.getGhostStates()
        #print(newGhostStates)
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print(newScaredTimes)

        "*** YOUR CODE HERE ***"
        #Let's discuss some things:
        #   -Function returns a number, where higher numbers are better
        #   -a food close to PACMAN should increase the number
        #   -a ghost close to PACMAN should decrease the number
        #   -an edible ghost located where the nextAction of PACMAN is,should critically increase the number 

        #So, all the above will be the parameters for the final value of the number that the function returns

        from util import manhattanDistance #as i did in the previous project (Project 1) , I will use manhattanDistance to represent the distances between states

        #closest food parameter
        closest_food_distance=None

        for food in newFood.asList():
            
            distance = manhattanDistance(newPos,food)

            if closest_food_distance==None:
                closest_food_distance = distance
            elif distance < closest_food_distance:
                closest_food_distance = distance

        if closest_food_distance==None: #no food left in this 
            food_score=-1
        else:
            food_score = 1.5/closest_food_distance #weight of food = 1.5

        
        #find closest ghost
        closest_ghost_distance=None

        for ghost in newGhostStates:
            
            distance = manhattanDistance(newPos,ghost.getPosition())

            if closest_ghost_distance==None:
                closest_ghost_distance = distance
            elif distance < closest_ghost_distance:
                closest_ghost_distance = distance


        if closest_ghost_distance == 0: #if the nextAction state is the nextGhost state
            
            if newScaredTimes[0] != 0: #if the ghost is edible | All the ghost have the same scared times so i just get the first element of newScaredTimes
                edible_ghost_score = 3     #weight of edible ghost = 3
                ghost_score = 0
            else: #if not edible
                edible_ghost_score = 0
                ghost_score = 3 # We really want to avoid this ghost
        
        
        else: #if the nextAction state is NOT the nextGhost state 

            if newScaredTimes[0] != 0: #if the ghost is edible | All the ghost have the same scared times so i just get the first element of newScaredTimes
                edible_ghost_score = 3     #weight of edible ghost = 3
                ghost_score = 0
            else: #if not edible
                edible_ghost_score = 0     
                ghost_score = 2 / closest_ghost_distance    #weight of inedible ghost = 2 | We really want to avoid ghosts
                #oso pio makria einai ena ghost,toso megalutero to closest_ghost_distance ara toso mikrotero to ghost_score 
            

        return successorGameState.getScore() + food_score + edible_ghost_score - ghost_score
         

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
        def my_minimax(gameState, agent, depth):
            #Opws ziteitai stin ekfwnisi prokeitai gia ena recursive function
            
            #Terminal condition
            if((self.depth == depth) or gameState.isLose() or gameState.isWin()): #if you won/lost/or reached the max depth of the game

                return [-1, self.evaluationFunction(gameState)]
            
            #agent has values (0,1,2....,n)
            #gameState.getNumAgents() returns values apo 2 (an exw dyo agents,ara agents pairnei times sto [0,1]) mexri m = n+1 (ara agents pairnei times sto [0,n] == [0,m-1])
            #Epishs,oi agents eksetazontai seiriaka
            #one pacman and ghosts moves are 1 depth

            agents_number = gameState.getNumAgents() - 1 # m-1

            if agent == self.index: #Ean eksetazw pacman,kai eimai edw,den exw terminal condition ara synexizw pros ta katw
                
                next_agent = agent+1    #oi agents eksetazontai seiriaka
            else: #ean eksetazw ghost
                #apla tsekarw kai gia to epomeno ghost, ektos an ta exw dei ola,ara irthe i ora gia pacman
                
                if agent != agents_number:    #den eimais to m-1 agent ara sto teleutaio agent
                    next_agent = agent+1
                else:   #exw dei ola ta ghosts,wra gia PACMAN pali
                    #increment depth
                    depth+=1
                    next_agent = self.index



            solution = [] #this will finally keep the best move for the agent we are on right now and it's score solution = ['Right', -3]

            for action in gameState.getLegalActions(agent): #for every possible outcome
                #print(action)
                
                if  len(solution)==0: #if this is the first loop

                    temp, score = my_minimax( gameState.generateSuccessor(agent, action), next_agent, depth ) #go down on the next agent (imagine that this recursion will work pretty much like a DFS)
                    solution.append(action)
                    solution.append(score)

                else:
         

                    temp, new_score = my_minimax( gameState.generateSuccessor(agent, action), next_agent, depth )

                    #if agent is PACMAN ( agent==0 ) we want the maximum score
                    if agent==self.index: 
                        if new_score > solution[1]:
                            solution[0] = action
                            solution[1] = new_score
                            
                    else:   #if agent is a ghost ( agent!=0 ) then we want the minimum score
                        if new_score < solution[1]:
                            solution[0] = action
                            solution[1] = new_score


            #print(solution[0])
            return solution

        action, score = my_minimax(gameState, self.index, 0) 
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def maximum_value(gameState, agent, depth, a, b):      #this will be called ONLY for PACMAN

            #check if this is terminal state
            #Terminal condition
            if((self.depth == depth) or gameState.isLose() or gameState.isWin()): #if you won/lost/or reached the max depth of the game

                return [-1, self.evaluationFunction(gameState)]

            next_agent = agent+1 #ghosts will run through min-value function so no need for more checks
            
           

            final=[]

            for action in gameState.getLegalActions(agent):

                

                #only ghosts can be next_agents,cause only pacman enters the max-value function
                if len(final)==0:  #if this is the first loop
                    
                    #get successor state for this action
                    next_state = gameState.generateSuccessor(agent, action)

                    v_new = minimum_value(next_state, next_agent, depth, a, b)

                    final.append(action)
                    final.append(v_new[1])

                    a=max(v_new[1], a)        #change a if needed

                else:       #not the first loop

                    if final[1] > b:
                        return final


                    #get successor state for this action
                    next_state = gameState.generateSuccessor(agent, action)

                    v_new = minimum_value(next_state, next_agent, depth, a, b)

                    if v_new[1] > final[1]: #if v' > v
                        final[0] = action
                        final[1] = v_new[1]
                        a=max(final[1], a)        #change a if needed
                        

            return final

        

        def minimum_value(gameState, agent, depth, a, b):

            #check if this is terminal state
            #Terminal condition
            if((self.depth == depth) or gameState.isLose() or gameState.isWin()): #if you won/lost/or reached the max depth of the game

                return [-1, self.evaluationFunction(gameState)]


            agents_number = gameState.getNumAgents() - 1 # m-1

            if agent != agents_number:  #den exw dei ola ta ghost
                next_agent = agent+1
            else:   #wra gia pacman pali
                next_agent = self.index
                depth += 1

            # v = +oo

            final = []

            for action in gameState.getLegalActions(agent):

                if len(final)==0:
                    #get successor state for this action
                    next_state = gameState.generateSuccessor(agent, action)

                    if next_agent == self.index: #an pame gia pacman
                        v_new = maximum_value(next_state, next_agent, depth, a, b)
                    else:   # next agent is ghost
                        v_new = minimum_value(next_state, next_agent, depth, a, b)


                    
                    final.append(action)
                    final.append(v_new[1])

                    b = min(v_new[1], b)        #change b if needed
            
                else:


                    if final[1] < a:
                        return final


                    #get successor state for this action
                    next_state = gameState.generateSuccessor(agent, action)

                    if next_agent == self.index: #an pame gia pacman
                        v_new = maximum_value(next_state, next_agent, depth, a, b)
                    else:   # next agent is ghost
                        v_new = minimum_value(next_state, next_agent, depth, a, b)


                    if v_new[1] < final[1]:    #if v' <  v
                        final[0] = action
                        final[1] = v_new[1]


                    b = min(v_new[1], b)    #change b if needed


            return final


        final = maximum_value(gameState, self.index, 0 , -float("inf"), float("inf"))
        return final[0]



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
        #same with minimax with some changes on what min agents(ghosts) chose

        def expectimax(gameState, agent, depth):
            #Opws ziteitai stin ekfwnisi prokeitai gia ena recursive function
            


            #Terminal condition
            if((self.depth == depth) or gameState.isLose() or gameState.isWin()): #if you won/lost/or reached the max depth of the game

                return [-1, self.evaluationFunction(gameState)]
            
            #agent has values (0,1,2....,n)
            #gameState.getNumAgents() returns values apo 2 (an exw dyo agents,ara agents pairnei times sto [0,1]) mexri m = n+1 (ara agents pairnei times sto [0,n] == [0,m-1])
            #Epishs,oi agents eksetazontai seiriaka
            #one pacman and ghosts moves are 1 depth

            agents_number = gameState.getNumAgents() - 1 # m-1

            if agent == self.index: #Ean eksetazw pacman,kai eimai edw,den exw terminal condition ara synexizw pros ta katw
                
                next_agent = agent+1    #oi agents eksetazontai seiriaka
            else: #ean eksetazw ghost
                #apla tsekarw kai gia to epomeno ghost, ektos an ta exw dei ola,ara irthe i ora gia pacman
                
                if agent != agents_number:    #den eimais to m-1 agent ara sto teleutaio agent
                    next_agent = agent+1
                else:   #exw dei ola ta ghosts,wra gia PACMAN pali
                    #increment depth
                    depth+=1
                    next_agent = self.index



            solution = [] #this will finally keep the best move for the agent we are on right now and it's score solution = ['Right', -3]

            available_actions_number = len(gameState.getLegalActions(agent))

            for action in gameState.getLegalActions(agent): #for every possible outcome
                #print(action)
                
                if  len(solution)==0: #if this is the first loop

                    temp, score = expectimax( gameState.generateSuccessor(agent, action), next_agent, depth ) #go down on the next agent (imagine that this recursion will work pretty much like a DFS)

                    if agent == self.index: #for PACMAN
                        
                        solution.append(action)
                        solution.append(score)
                    else:   #for ghosts
                        #oi expected times tou ghost (min agent) einai o mesos oros olwn twn dynatwn apotelesmatwn gia kathe kinhsh tou
                        solution.append(action)
                        solution.append(score/available_actions_number)
                else:
         

                    temp, new_score = expectimax( gameState.generateSuccessor(agent, action), next_agent, depth )

                    #if agent is PACMAN ( agent==0 ) we want the maximum score
                    if agent==self.index: 
                        if new_score > solution[1]:
                            solution[0] = action
                            solution[1] = new_score
                            
                    else:   #if agent is a ghost ( agent!=0 ) 
                        #oi expected times tou ghost (min agent) einai o mesos oros olwn twn dynatwn apotelesmatwn gia kathe kinhsh tou
                        
                        solution[0] = action
                        solution[1] += new_score/available_actions_number    #construct the sum of scores for this min agent


            #print(solution[0])

            
            
            return solution

        action, score = expectimax(gameState, self.index, 0) 
        return action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)

    #print(action)
    #print('\nsuccessor:\n', successorGameState,'\n action:', action, '\ncurrent:\n', currentGameState)
    pacPos = currentGameState.getPacmanPosition()
    #print(pacPos)
    foodPos = currentGameState.getFood()
    #print(foodPos)
    #print(foodPos.asList())
    GhostStates = currentGameState.getGhostStates()
    #print(GhostStates)
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    #print(ScaredTimes)

    "*** YOUR CODE HERE ***"
    #Let's discuss some things:
    #   -Function returns a number, where higher numbers are better
    #   -a food close to PACMAN should increase the number
    #   -a ghost close to PACMAN should decrease the number
    #   -an edible ghost located where the nextAction of PACMAN is,should critically increase the number 

    #So, all the above will be the parameters for the final value of the number that the function returns

    from util import manhattanDistance #as i did in the previous project (Project 1) , I will use manhattanDistance to represent the distances between states

    #closest food parameter
    closest_food_distance=None

    for food in foodPos.asList():
            
        distance = manhattanDistance(pacPos,food)

        if closest_food_distance==None:
            closest_food_distance = distance
        elif distance < closest_food_distance:
            closest_food_distance = distance

    if closest_food_distance==None: #no food left in this 
        food_score=-1
    else:
        food_score = 1.5/closest_food_distance #weight of food = 1.5

        
    #find closest ghost
    closest_ghost_distance=None

    for ghost in GhostStates:
            
        distance = manhattanDistance(pacPos,ghost.getPosition())

        if closest_ghost_distance==None:
            closest_ghost_distance = distance
        elif distance < closest_ghost_distance:
            closest_ghost_distance = distance


    if closest_ghost_distance == 0: #if the pacman position is a ghost position
            
        if ScaredTimes[0] != 0: #if the ghost is edible | All the ghost have the same scared times so i just get the first element of newScaredTimes
            edible_ghost_score = 3     #weight of edible ghost = 3
            ghost_score = 0
        else: #if not edible
            edible_ghost_score = 0
            ghost_score = 3 # We really want to avoid this ghost
        
        
    else: #if the pacman postition is NOT a ghost position 

        if ScaredTimes[0] != 0: #if the ghost is edible | All the ghost have the same scared times so i just get the first element of newScaredTimes
            edible_ghost_score = 3     #weight of edible ghost = 3
            ghost_score = 0
        else: #if not edible
            edible_ghost_score = 0     
            ghost_score = 2 / closest_ghost_distance    #weight of inedible ghost = 2 | We really want to avoid ghosts
            #oso pio makria einai ena ghost,toso megalutero to closest_ghost_distance ara toso mikrotero to ghost_score 
            

    return currentGameState.getScore() + food_score + edible_ghost_score - ghost_score
    

# Abbreviation
better = betterEvaluationFunction
