import math
import random

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                EMULATION OF REAL VALUES IN A SOL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def emulation(sol, routeMaxCost, seed):
    # This code assumes binary random rewards on each node 
    for route in sol.routes:
        routeReward = 0 # route reward in this run
        routeCost = 0 # time- or distance-based cost
        for e in route.edges:
            node = e.end # end node of the edge
            # DEHTOP mod ---
            # routeReward += node.reward
            drone = route.drone_type
            weather = get_conditions(node, seed)
            routeReward += getBinaryRandomReward(drone, weather, node)
            edgeCost = e.cost
           # ---------------
            routeCost += edgeCost
        
        if routeCost > routeMaxCost: # violates constraint on max cost
            routeReward = 0 # penalty for violating the max time allowed per route

    sol.reward_sim = routeReward


# DEHTOP mod ---
def getBinaryRandomReward(drone, weather, node, seed=0, verbose=False):
    """ Generates the binary random reward for node based on conditions """
    #TODO: use the global seed to always generate the same output?
    b_0, b_d, b_w = getCoeficients(node, drone)
    linearCombination = b_0 + b_d + b_w * weather
    prob = 1 / (1 + math.exp(-linearCombination))
    # if seed != 0:
    # random.seed(seed)
    [reward] = random.choices([0,1],[1-prob, prob])
    if verbose:
        print(f"seed: {seed}; probability: {prob}; reward: {reward}")
    return reward

def getCoeficients(node, drone):
    """ Obtains the coeficientes for a given node and drone """
    # TODO: look in fixed table given the node and the drone
    b_0 = 0.1
    b_d = 0.1
    b_w = 1.0
    return b_0, b_d, b_w

def get_conditions(node, seed):
    #TODO: obtain weather conditions for instance seed at given node
    # random:
    # new_seed = str(node.ID) + str(seed)
    # random.seed(new_seed)
    [weather] = random.choices([0,1],[0.5,0.5])
    return weather

# --------------
if __name__ == "__main__":
    r = getBinaryRandomReward(drone="A",weather=0,node=3,seed=1, verbose=True)
    
    # Test for random.choices
    # a = []
    # for i in range(1000):
    #     a.append(random.choices([0,1],[0.1,0.9]))
    # print(f"unos: {a.count([1])}")
    # print(f"ceros: {a.count([0])}")
