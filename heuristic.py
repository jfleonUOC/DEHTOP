import time
import copy
import math
import random
import operator
import numpy as np

from aux_objects import Edge, Route, Solution
from aux_functions import read_tests, read_instance, printRoutes
from reinflearn import Agent, train_agent, generate_historical_data
from emulation import emulation, get_conditions

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    MULTI-START FRAMEWORK
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    GENERATE SAVINGS LIST
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def generateSavingsList(nodes):
    """
    Generate savings list of nodes: construct edges and savings from nodes
    For the DEHTOP the Efficiency List depends on the weather and type of drone
    """
    start = nodes[0]
    finish = nodes[-1]
    for node in nodes[1:-1]: # excludes the start and finish depots
        snEdge = Edge(start, node) # creates the (start, node) edge (arc)
        nfEdge = Edge(node, finish) # creates the (node, finish) edge (arc)
        # compute the Euclidean distance as cost
        snEdge.cost = math.sqrt((node.x - start.x)**2 + (node.y - start.y)**2)
        nfEdge.cost = math.sqrt((node.x - finish.x)**2 + (node.y - finish.y)**2)
        # save in node a reference to the (depot, node) edge (arc)
        node.dnEdge = snEdge
        node.ndEdge = nfEdge

    savingsList = []
    for i in range(1, len(nodes) - 2): # excludes the start and finish depots
        iNode = nodes[i]
        for j in range(i + 1, len(nodes) - 1):
            jNode = nodes[j]
            ijEdge = Edge(iNode, jNode) # creates the (i, j) edge
            jiEdge = Edge(jNode, iNode)
            ijEdge.invEdge = jiEdge # sets the inverse edge (arc)
            jiEdge.invEdge = ijEdge
            # compute the Euclidean distance as cost
            ijEdge.cost = math.sqrt((jNode.x - iNode.x)**2 + (jNode.y - iNode.y)**2)
            jiEdge.cost = ijEdge.cost # assume symmetric costs
            # compute efficiency as proposed by Panadero et al.(2020)
            # DEHTOP mod ---
            ijSavings = iNode.ndEdge.cost + jNode.dnEdge.cost - ijEdge.cost
            ijEdge.savings = ijSavings
            jiSavings = jNode.ndEdge.cost + iNode.dnEdge.cost - jiEdge.cost
            jiEdge.savings = jiSavings
            # save both edges in the savings list
            savingsList.append(ijEdge)
            savingsList.append(jiEdge)

    # sort the list of edges from higher to lower savings
    savingsList.sort(key = operator.attrgetter("savings"), reverse = True)
    return savingsList

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    GENERATE DUMMY SOLUTION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def dummySolution(routeMaxCost, nodes):
    """ Generate Dummy Solution """
    #TODO: add "n_aerial" and "fleetSize" as function parameters
    sol = Solution()
    for node in nodes[1:-1]: # excludes the start and finish depots
        snEdge = node.dnEdge
        nfEdge = node.ndEdge
        snfRoute = Route() # construct the route (start, node, finish)
        snfRoute.edges.append(snEdge)
        snfRoute.reward += node.reward
        snfRoute.cost += snEdge.cost
        snfRoute.edges.append(nfEdge)
        snfRoute.cost += nfEdge.cost
        # DEHTOP mod ----
        #TODO: assign drone_type to each route based on the instance information
        # prob = n_aerial/fleetSize
        # snfRoute.drone_type = random.choices(["A","M"],[prob, 1-prob])
        # ---------------
        node.inRoute = snfRoute # save in node a reference to its current route
        node.isLinkedToStart = True # this node is currently linked to start depot
        node.isLinkedToFinish = True # this node is currently linked to finish depot
        if snfRoute.cost <= routeMaxCost:
            sol.routes.append(snfRoute) # add this route to the solution
            sol.cost += snfRoute.cost
            sol.reward += snfRoute.reward # total reward in route

    return sol

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            MERGING PROCESS IN THE PJ'S HEURISTIC
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def compute_sol(fleetSize, routeMaxCost, nodes, sav_list, agent, seed):
    """ Perform the BR edge-selection & routing-merging iterative process """
    sol = dummySolution(routeMaxCost, nodes) # compute the dummy solution
    #TODO: create an "efficiency list" that allows looping
    savList = copy.copy(sav_list) # make a shallow copy of the savings list since it will be modified
    while len(savList) > 0: # list is not empty
        # DEHTOP mod ---
        # The merging node is selected based on the reinf. learning algorithm
        # instead of a on biased-randomized position of an ordered efficiency list
        # recalculate effList
        print(f"* Efficiency list lenght: {len(savList)} *")
        if agent == None: # use savings
            ijEdge = savList[0]
        else: # use RL agent
            ijEdge = select_edge(savList, agent, seed)
        print(f"> merging {ijEdge}")
        # i_node = agent.select_action()
        # j_node = agent.select_action()
        # ijEdge = savList.pop(position) # select the next edge from the list
        ijEdge_idx = savList.index(ijEdge)
        savList.pop(ijEdge_idx)
        # --------------
        sol, mergeOk = merge_edge(sol, ijEdge, routeMaxCost)
        if mergeOk: print("> merge OK !!")
        if not mergeOk:
        # if still in list, delete edge (j, i) since it will not be used
            jiEdge = ijEdge.invEdge
            if jiEdge in savList:
                savList.remove(jiEdge)
        # printRoutes(sol)

    # sort the list of routes in sol by reward (reward) and delete extra routes
    sol.routes.sort(key = operator.attrgetter("reward"), reverse = True)
    for route in sol.routes[fleetSize:]:
        sol.reward -= route.reward # update reward
        sol.cost -= route.cost # update cost
        sol.routes.remove(route) # delete extra route
    #TODO: make sure that the drone type number constrain is respected!

    return sol

def merge_edge(sol, ijEdge, routeMaxCost):
    # determine the nodes i < j that define the edge
    iNode = ijEdge.origin
    jNode = ijEdge.end
    # determine the routes associated with each node
    iRoute = iNode.inRoute
    jRoute = jNode.inRoute
    # check if merge is possible
    isMergeFeasible = checkMergingConditions(iNode, jNode, iRoute, jRoute, ijEdge, routeMaxCost)
    # if all necessary conditions are satisfied, merge and delete edge (j, i)
    if isMergeFeasible == True:
        # iRoute will contain edge (i, finish)
        iEdge = iRoute.edges[-1] # iEdge is (i, finish)
        # remove iEdge from iRoute and update iRoute cost
        iRoute.edges.remove(iEdge)
        iRoute.cost -= iEdge.cost
        # node i will not be linked to finish depot anymore
        iNode.isLinkedToFinish = False
        # jRoute will contain edge (start, j)
        jEdge = jRoute.edges[0]
        # remove jEdge from jRoute and update jRoute cost
        jRoute.edges.remove(jEdge)
        jRoute.cost -= jEdge.cost
        # node j will not be linked to start depot anymore
        jNode.isLinkedToStart = False
        # add ijEdge to iRoute
        iRoute.edges.append(ijEdge)
        iRoute.cost += ijEdge.cost
        iRoute.reward += jNode.reward
        jNode.inRoute = iRoute
        # add jRoute to new iRoute
        for edge in jRoute.edges:
            iRoute.edges.append(edge)
            iRoute.cost += edge.cost
            iRoute.reward += edge.end.reward
            edge.end.inRoute = iRoute
        # delete jRoute from emerging solution
        sol.cost -= ijEdge.savings
        sol.routes.remove(jRoute)
    return sol, isMergeFeasible

def checkMergingConditions(iNode, jNode, iRoute, jRoute, ijEdge, routeMaxCost, verbose = True):
    """ Check if merging conditions are met """
    # condition 1: iRoute and jRoute are not the same route object
    if iRoute == jRoute:
        if verbose: print("> cannot merge: iRoute and jRoute are the same route object")
        return False
    # condition 2: jNode has to be linked to start and i node to finish
    if iNode.isLinkedToFinish == False or jNode.isLinkedToStart == False:
        if verbose: print("> cannot merge: jNode has to be linked to start and i node to finish")
        return False
    # condition 3: cost after merging does not exceed maxTime (or maxCost)
    if iRoute.cost + jRoute.cost - ijEdge.savings > routeMaxCost:
        if verbose: print("> cannot merge: cost after merging exceeds maxTime (or maxCost)")
        return False
    # DEHTOP mod ---
    # condition 4: check battery status
    #TODO: the cost already represent the battery status?
    # condition 5: check that both routes use the same type of drone
    #TODO: modify routes using "assign_routes(iRoute,jRoute)"
    # --------------
    # else, merging is feasible
    return True

def select_edge(savings_list, rl_agent, inst_seed, alg = 1):
    """ Select edge to merge based on (alg=1) epsilon-greedy or (alg=2) Upper Confidence Bounds """
    efficiency_list = compute_efficiency(savings_list, rl_agent, inst_seed)
    if alg == 1: # 1. Epsilon-greedy
        edge = epsilonGreedy(efficiency_list)
    else: # 2. Upper Confidence Bounds
        edge = UBC(efficiency_list)
    return edge

def compute_efficiency(savings_list, rl_agent, inst_seed, alpha=0.5):
    eff_list = copy.copy(savings_list)
    for edge in eff_list:
        # determine the nodes i < j that define the edge
        iNode = edge.origin
        jNode = edge.end
        # determine the routes associated with each node
        iRoute = iNode.inRoute
        jRoute = jNode.inRoute
        # determine dynamic conditions for each node
        i_weather = get_conditions(iNode, inst_seed)
        j_weather = get_conditions(jNode, inst_seed)
        #TODO: check that drone type is (1) the same in both routes, (2) allowed - number of drones is limited
        iRoute, jRoute = assign_drones(iRoute, jRoute)
        drone = iRoute.drone_type
        # determine rewards
        # print(f"node: {iNode}, drone: {drone}, weather: {i_weather}")
        # print(f"node: {jNode}, drone: {drone}, weather: {j_weather}")
        iNode.reward = rl_agent.get_estimated_reward(iNode, drone, i_weather)
        jNode.reward = rl_agent.get_estimated_reward(jNode, drone, j_weather)
        edgeReward = iNode.reward + jNode.reward
        # compute efficiency
        #TODO: scale up the edgeReward component, which is <= 1
        edge.efficiency = alpha * edge.savings + (1 - alpha) * edgeReward * 10
    # sort the list of edges from higher to lower efficiency
    eff_list.sort(key = operator.attrgetter("efficiency"), reverse = True)

    return eff_list

def epsilonGreedy(eff_list, epsilon = 0.1):
    random.seed(None) # initialize the random seed for the purpose of RL agent selection
    if random.random() < epsilon:
        # Explore: Select a random edge
        action = random.choice(eff_list)
    else:
        # Exploit: Select the edge with the highest efficiency
        action = eff_list[0]

    return action

def UBC(eff_list):
    random.seed(None) # initialize the random seed for the purpose of RL agent selection
    pass

def assign_drones(route1, route2):
    """
    Function to assign the correct drone type to each route before merging
    """
    if route1.drone_type != route2.drone_type: 
        if route1.drone_type == None: # means that the other is already assigned
            route1.drone_type = route2.drone_type
        elif route2.drone_type == None: # means that the other is already assigned
            route2.drone_type = route1.drone_type
        else: # both are not None but different -> longer route wins; if equals, route1 wins
            if len(route1.edges) >= (route2.edges):
                route2.drone_type = route1.drone_type
            else:
                route1.drone_type = route2.drone_type
    elif route1.drone_type == None: # both are None -> assign "A" (aerial) by default
        route1.drone_type = "A"
        route2.drone_type = "A"
    # if they are equal and not None, simply return the unmodified routes
    return route1, route2

if __name__ == "__main__":
    # read instance data
    print("*** READING DATA ***")
    file_name = r"data2/p1.2.a.txt"
    seed = 3 #TODO: to be read from the instance or from the test
    fleetSize, routeMaxCost, nodes = read_instance(file_name)

    # create RL agent
    agent = Agent(nodes)
    # print(agent.act_table)
    print("*** SIMULATING DATA ***")
    historical_data = generate_historical_data(100, nodes, seed)
    print("*** TRAINING AGENT ***")
    train_agent(agent, historical_data)
    print(agent.act_table)

    print("*** RUNNING HEURISTIC ***")
    savings_list = generateSavingsList(nodes)
    # print(savings_list)

    # compute benchmark
    print("*** COMPUTING BENCHMARK ***")
    benchmark = compute_sol(fleetSize, routeMaxCost, nodes, savings_list, None, seed)   

    # compute solution
    print("*** COMPUTING SOLUTION ***")
    sol = compute_sol(fleetSize, routeMaxCost, nodes, savings_list, agent, seed)   

    # emulate proposed solutions
    print("*** RESULTS BENCHMARK ***")
    printRoutes(benchmark)
    emulation(benchmark, routeMaxCost, seed)
    print(benchmark)
    print("*** RESULTS SOLUTION ***")
    printRoutes(sol)
    emulation(sol, routeMaxCost, seed)
    print(sol)
