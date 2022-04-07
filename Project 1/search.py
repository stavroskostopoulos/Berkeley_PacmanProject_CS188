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
    """
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    
    #x = problem.getSuccessors(problem.getStartState())
    #print(x[1][0])
    #print("Next successors: ", problem.getSuccessors(x[1][0]))


    "*** YOUR CODE HERE ***"

    start = problem.getStartState() #get the start state of the search problem
    currentState = start #the name speaks for itself

    stoiva = util.Stack()#the queue we are going to use for our DFS implementation
    visited = set() #this set will keep the states we have visited O(1)
    path = []   #our queue elements will be tuples like this one:    ( (4,5), path ) where path is a list of the directions we used to get in this element ex.: ['South','West',...]

    visited.add(start)
    
    while problem.isGoalState(currentState)==False:
        
        successors = problem.getSuccessors(currentState)
        visited.add(currentState)    #add this state to visited
        
        for tripleta in successors:

            if tripleta[0] in visited: #ignore
                continue
            else:
                
                newpath = path.copy() #in newpath variable we construct the path we will save in the queue for this new neighbour(tripleta[0])
                
                newpath.append(tripleta[1]) #append the directions for this new state to the newpath
                
                stoiva.push( (tripleta[0], newpath) )    #push the new path

        #endfor

        nextone = stoiva.pop()#get the next state we are visiting

        currentState = nextone[0]   #first element of the tuple ex.: (4,5)
        
        path = nextone[1]           #path for the next one

    #endwhile

    
    #currentState is the goal state right now and path is the path


    return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
   
    "*** YOUR CODE HERE ***"

    start = problem.getStartState() #get the start state of the search problem
    currentState = start #the name speaks for itself

    queue = util.Queue()#the queue we are going to use for our BFS implementation
    visited = set() #this set will keep the states we have visited O(1)
    path = []   #our queue elements will be tuples like this one:    ( (4,5), path ) where path is a list of the directions we used to get in this element ex.: ['South','West',...]

    visited.add(start)
    
    while problem.isGoalState(currentState)==False:
        #print(currentState)

        successors = problem.getSuccessors(currentState)
        
        for tripleta in successors:

            if tripleta[0] in visited: #ignore
                continue
            else:
                visited.add(tripleta[0])    #add this neighbour to visited
                newpath = path.copy() #in newpath variable we construct the path we will save in the queue for this new neighbour(tripleta[0])
                newpath.append(tripleta[1]) #append the directions for this new state to the newpath
                queue.push( (tripleta[0], newpath) )    #push the new path

        #endfor

        nextone = queue.pop()#get the next state we are visiting

        currentState = nextone[0]   #first element of the tuple ex.: (4,5)
        path = nextone[1]           #path for the next one

    #endwhile

    #currentState is the goal state right now and path is the path


    return path



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    currentState = start
    
    pqueue = util.PriorityQueue()   #the elements we push to the priority queue will be triples like this one:    ( (4,5), path, successors ) where path is a list of the directions we used to get in this element ex.: ['South','West',...] 

    path = []

    visited = set()

    while problem.isGoalState(currentState)==False:

        

        

        if currentState not in visited:#don't expand nodes twice
            successors = problem.getSuccessors(currentState)
            visited.add(currentState) #don't add the same state to visited twice
        #else do nothing we already saved the successors from the queue pop of the previous loop
            
        

        for tripleta in successors:

            if tripleta[0] in visited:  #skip if visited
                continue
            else:   #if not visited
                    
                newpath = path.copy()#in newpath variable we construct the path we will save in the queue for this new neighbour(tripleta[0])

                newpath.append(tripleta[1])#append the directions for this new state to the newpath

                pqueue.push( (tripleta[0], newpath, successors), problem.getCostOfActions(newpath) )

        
        #endfor

        nextone = pqueue.pop()  #get the state with the least cost |returs the triple ex.: ( (4,5), path, successors )

        currentState = nextone[0]
        path = nextone[1]
        successors = nextone[2]

    #endwhile


    return path


    
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    currentState = start
    
    pqueue = util.PriorityQueue()   #the elements we push to the priority queue will be triples like this one:    ( (4,5), path, successors ) where path is a list of the directions we used to get in this element ex.: ['South','West',...] 

    path = []

    visited = set()

    visited_astar_scores = {}


    while problem.isGoalState(currentState)==False:

    

        if currentState not in visited:#don't expand nodes twice
            successors = problem.getSuccessors(currentState)
            visited.add(currentState) #don't add the same state to visited twice
            visited_astar_scores[currentState] = problem.getCostOfActions(path)+heuristic(currentState, problem) #save the A* score of this state to our dictionary
            #print(problem.getCostOfActions(path))
        #else do nothing we already saved the successors from the queue pop of the previous loop
        else:#if we have visited this state again, check whether the A* score now is less than the one we have visited

            new_score = problem.getCostOfActions(path)+heuristic(currentState, problem)    
            
            #it is visited so it's saved in visited set and  visited_astar_scores dictionary | So there's no chance of KeyError

            if visited_astar_scores[currentState] < new_score:  #if previous score less than the new one, there's no need to check this state

                #get next state
                nextone = pqueue.pop()  #get the state with the least cost |returs the triple ex.: ( (4,5), path, successors )

                currentState = nextone[0]
                path = nextone[1]
                successors = nextone[2]

                #and try again
                continue

            else:   #if new score is less than the previous one

                #just update the A* score dictionary value for this state
                visited_astar_scores[currentState] = problem.getCostOfActions(path)+heuristic(currentState, problem)

        #endif

        for tripleta in successors:

            if tripleta[0] in visited:  #skip if visited
                continue
            else:   #if not visited
                    
                newpath = path.copy()#in newpath variable we construct the path we will save in the queue for this new neighbour(tripleta[0])

                newpath.append(tripleta[1])#append the directions for this new state to the newpath

                pqueue.push( (tripleta[0], newpath, successors), problem.getCostOfActions(newpath)+heuristic(tripleta[0], problem) )

        
        #endfor

        nextone = pqueue.pop()  #get the state with the least cost |returs the triple ex.: ( (4,5), path, successors )

        currentState = nextone[0]
        path = nextone[1]
        successors = nextone[2]

    #endwhile
    
    return path



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
