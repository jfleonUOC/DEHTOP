import math
import random

from aux_objects import Node

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                EMULATION OF REAL VALUES IN A SOL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def emulation(sol, routeMaxCost, seed, print_results=True):
    # This code assumes binary random rewards on each node 
    sol.reward_sim = 0
    for route in sol.routes:
        routeReward = 0 # route reward in this run
        routeCost = 0 # time- or distance-based cost
        for e in route.edges[:-1]:
            node = e.end # end node of the edge
            # drone = route.drone_type
            weather = get_conditions(node, seed)
            # reward = getBinaryRandomReward(drone, weather, node, seed=seed)
            reward = getBinaryRandomReward(node, weather, seed=seed)
            node.realReward = reward
            routeReward += reward
            # if reward != 0: print(f"node with reward: {node} ; drone = {drone}, weather = {weather}")
            # if reward != 0: print(f"node with reward: {node} ; weather = {weather}")
            edgeCost = e.cost
            routeCost += edgeCost
        
        if routeCost > routeMaxCost: # violates constraint on max cost
            print(f"Route exceded max cost! route cost: {routeCost}; max. cost: {routeMaxCost}; difference: {routeCost-routeMaxCost}")
            routeReward = 0 # penalty for violating the max time allowed per route

        sol.reward_sim += routeReward
    
    if print_results: printEmuRoutes(sol)


# DEHTOP mod ---
# def getBinaryRandomReward(drone, weather, node, seed, verbose=False):
def getBinaryRandomReward(node, weather, seed, verbose=False):
    """ Generates the binary random reward for node based on conditions """
    #TODO: use the global seed to always generate the same output
    # b_0, b_d, b_w = getCoeficients(node, drone)
    b_0, b_d, b_w = getCoeficients(node)
    # print(f"{b_0 =}, {b_d =}, {b_w =}")
    linearCombination = b_0 + b_d + b_w * weather
    prob = 1 / (1 + math.exp(-linearCombination))
    # random.seed(seed + drone + str(weather) + str(node))
    random.seed(seed + str(weather) + str(node))
    [reward] = random.choices([0,1],[1-prob, prob])
    if verbose:
        print(f"seed: {seed}; probability: {prob}; reward: {reward}")
    return reward

def getCoeficients(node):
    """ Obtains the coeficientes for a given node """
    # TODO: look in fixed table given the node
    b_0 = 0.0
    b_d = 0.0
    b_w = 2.0
    return b_0, b_d, b_w

def get_conditions(node, seed):
    # Obtain weather conditions for instance seed at given node
    # conditions: wind, waves, visibility, rain
    random.seed(seed + str(node.ID))
    [weather] = random.choices([0,1],[0.5,0.5])
    return weather

def printEmuRoutes(sol):
    """ Print routes in a solution """
    print(f"*SOLUTION ROUTES*")
    for route in sol.routes:
        print("0", end = "")
        for e in route.edges:
            if e.end.realReward != 0:
                print(f"->{e.end.ID}*", end="")
            else:
                print(f"->{e.end.ID}", end="")
        print("\nRoute det reward:", route.reward, "; det cost:", route.cost)
    print(f"*SUMMARY*")
    print(f"routes: {len(sol.routes)} \ncost: {sol.cost}, real reward: {sol.reward_sim}")


# --------------
if __name__ == "__main__":
    node = Node(1,0,0,0)
    # r = getBinaryRandomReward(drone="A",weather=0,node=node.ID, seed=str(1), verbose=True)
    r = getBinaryRandomReward(weather=1,node=node.ID, seed=str(1), verbose=True)
    
    # Test for random.choices
    # a = []
    # for i in range(1000):
    #     a.append(random.choices([0,1],[0.1,0.9]))
    # print(f"unos: {a.count([1])}")
    # print(f"ceros: {a.count([0])}")
