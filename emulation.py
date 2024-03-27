import math
import random
import re

from aux_objects import Node

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                EMULATION OF REAL VALUES IN A SOL
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def emulation(sol, routeMaxCost, seed, print_results=True):
    # This code assumes binary random rewards on each node 
    sol.reward_sim = 0
    # print(f"emulation seed: {seed}")
    for route in sol.routes:
        routeReward = 0 # route reward in this run
        routeCost = 0 # time- or distance-based cost
        for e in route.edges[:-1]:
            node = e.end # end node of the edge
            weather = get_conditions(node.ID, seed)
            [wind_c, rain_c] = weather
            # print(f"node: {node}; weather: {weather}")
            # prob, reward = getBinaryRandomReward(node.ID, weather, seed=seed)
            prob, reward = getBinaryRandomReward(node_id=node.ID, wind=wind_c, rain=rain_c, seed=seed, verbose=False)
            node.realReward = reward
            node.probability = prob
            # print(f"node: {node}; reward: {node.realReward}; probability: {node.probability}")
            routeReward += reward
            # if reward != 0: print(f"node with reward: {node} ; weather = {weather}")
            edgeCost = e.cost
            routeCost += edgeCost
        
        if routeCost > routeMaxCost: # violates constraint on max cost
            print(f"Route exceded max cost! route cost: {routeCost}; max. cost: {routeMaxCost}; difference: {routeCost-routeMaxCost}")
            routeReward = 0 # penalty for violating the max time allowed per route

        sol.reward_sim += routeReward
    
    if print_results: printEmuRoutes(sol)


def getBinaryRandomReward_old(node_id, wind, rain, seed, verbose=False):
    """ Generates the binary random reward for node based on conditions """
    # create weather as a list with the different conditions
    weather = [wind, rain]
    num_coeffs = len(weather)
    coeffs = getCoefficients(node_id, seed, num_coeffs)
    # print(f"{coeffs =}")
    linearCombination = sum(coeffs[i]*weather[i] for i in range(num_coeffs))
    prob = 1 / (1 + math.exp(-linearCombination))
    weather_str = "".join(str(i) for i in weather) # string that represents weather conditions
    random.seed(seed + str(node_id) + weather_str)
    [reward] = random.choices([0,1],[1-prob, prob])
    if verbose:
        print(f"seed: {seed}; probability: {prob}; reward: {reward}")

    return prob, reward

def getBinaryRandomReward(node_id, wind, rain, seed, verbose=False):
    """ Generates the binary random reward for node based on conditions """
    # create weather as a list with the different conditions
    weather = [wind, rain]
    prob = prob_sum(wind, rain)
    weather_str = "".join(str(i) for i in weather) # string that represents weather conditions
    bin_seed = seed + str(node_id) + weather_str
    # print(f"getBinaryRandomReward seed: {bin_seed}")
    random.seed(bin_seed)
    [reward] = random.choices([0,1],[1-prob, prob])
    if verbose:
        print(f"seed: {seed}; probability: {prob}; reward: {reward}")

    return prob, reward

def prob_sum(wind, rain):
    """ Returns probability based on weather conditions """
    return {
        2 : 1.00,
        1 : 0.85,
        0 : 0.20,
        -1: 0.10,
        -2: 0.00,
    }[wind + rain]

def getCoefficients(node_id, seed, num_coeffs):
    """ Obtains the coefficients for a given node """
    # the coefficients are linked to the instance (not to the test or episode)
    instance_seed = re.match("^(.*?)txt",seed).group(0)
    random.seed(instance_seed+str(node_id))
    coeffs = []
    for _ in range(num_coeffs):
        coeffs.append(random.randint(0,1))

    return coeffs

def get_conditions(node_id, seed):
    """ Obtain weather conditions for instance seed at given node """
    # possible conditions: wind, waves, visibility, rain, temperature
    # columns_names = ["nodes", "wind", "rain", "visi", "temp", "value", "visited"]
    cond_seed = seed + str(node_id)
    # print(f"get_conditions seed: {cond_seed}")
    random.seed(cond_seed)
    wind = rain = random.randint(-1, 1)
    weather = [wind, rain]

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


if __name__ == "__main__":
    seed = "test_instance.tt4827h2"
    node = Node(ID=1, x=0, y=0, reward=0)
    weather = get_conditions(node.ID, seed)
    [wind_c, rain_c] = weather
    prob, reward = getBinaryRandomReward(node_id=node.ID, wind=wind_c, rain=rain_c, seed=seed, verbose=True)

    # seed = "test_instance.txt4827"
    # instance_seed = re.match("^(.*?)txt",seed).group(0)
    # print(instance_seed)
    
    # Test for random.choices
    # a = []
    # for i in range(1000):
    #     a.append(random.choices([0,1],[0.1,0.9]))
    # print(f"unos: {a.count([1])}")
    # print(f"ceros: {a.count([0])}")
