import time
import copy
import math
import random
import operator
from tqdm import tqdm
import numpy as np
from scipy import stats
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt

from aux_objects import Edge, Route, Solution
from aux_functions import read_tests, read_instance, printRoutes, graphRoutes
from reinflearn import Agent, train_agent, generate_historical_data, export_agent_to_file, import_agent_from_file, export_historical_data, import_historical_data
from emulation import emulation, get_conditions, getBinaryRandomReward

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    MULTI-START FRAMEWORK
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def multistart(file_name, n_episodes, alpha, alg, testSeed=str(0)):
    """ Online learning along N episodes """
    # read instance data
    print("*** READING DATA ***")
    # fleetSize, routeMaxCost, nodes = read_instance(file_name)
    # nodes_ben = copy.deepcopy(nodes)
    # nodes_sol = copy.deepcopy(nodes)
    fleetSize, routeMaxCost, nodes_ben = read_instance(file_name)
    fleetSize, routeMaxCost, nodes_sol = read_instance(file_name)

    # seeds: instance > test > episode
    instance_seed = file_name
    test_seed = instance_seed+testSeed #TODO: to be read from the test file: represents the variablity for that particular test

    # create RL agent
    agent = Agent(nodes_sol)
    # print(agent.act_table)

    print("*** RUNNING HEURISTIC ***")
    savings_list_ben = generateSavingsList(nodes_ben)
    savings_list_sol = generateSavingsList(nodes_sol)
    # print(savings_list)

    historical_data = []
    ben_episode_reward=[]
    sol_episode_reward=[]

    # compute benchmark
    for episode in tqdm(range(n_episodes)):
        # print(f"* COMPUTING EPISODE {episode} *")
        episode_seed = test_seed+str(episode)

        benchmark = compute_sol(fleetSize, routeMaxCost, nodes_ben, savings_list_ben, None, historical_data, episode_seed, alpha, verbose=False)   
        emulation(benchmark, routeMaxCost, episode_seed, print_results=False)

        # add historical data only for the visited nodes
        visited_nodes = get_visited_nodes([benchmark])
        new_historical_data = extract_hist_data(visited_nodes, episode_seed)
        historical_data.extend(new_historical_data)

        ben_episode_reward.append(benchmark.reward_sim)
        # print(f"lenght historical data: {len(historical_data)}")

    # compute solution
    for episode in tqdm(range(n_episodes)):
        # print(f"* COMPUTING EPISODE {episode} *")
        episode_seed = test_seed+str(episode)

        sol = compute_sol(fleetSize, routeMaxCost, nodes_sol, savings_list_sol, agent, historical_data, episode_seed, alpha, alg=alg, verbose=False)   
        emulation(sol, routeMaxCost, episode_seed, print_results=False)

        # add historical data only for the visited nodes
        visited_nodes = get_visited_nodes([sol])
        new_historical_data = extract_hist_data(visited_nodes, episode_seed)
        historical_data.extend(new_historical_data)

        # train agent
        train_agent(agent, new_historical_data, verbose=False) 

        sol_episode_reward.append(sol.reward_sim)
    
    # export online trained agent
    file_output = file_name[:-4] + "_online.csv"
    export_agent_to_file(agent, file_output)

    # plot cumulated reward    
    ben_cumulated_reward = np.cumsum(ben_episode_reward)
    sol_cumulated_reward = np.cumsum(sol_episode_reward)
    episodes = [i+1 for i in range(n_episodes)]
    # moving average
    size = int(round(n_episodes/10))
    ben_filtered_reward = uniform_filter1d(ben_episode_reward, size=size)
    sol_filtered_reward = uniform_filter1d(sol_episode_reward, size=size)

    fig, axs = plt.subplots(2)
    axs[0].plot(episodes,ben_cumulated_reward, color="green", label="benchmark")
    axs[0].plot(episodes,sol_cumulated_reward, color="red", label="RL agent")
    axs[0].legend(loc="upper left")
    axs[1].plot(episodes, ben_episode_reward, color="green", alpha=0.3)
    axs[1].plot(episodes, sol_episode_reward, color="red", alpha=0.3)
    axs[1].plot(episodes, ben_filtered_reward, color="green")
    axs[1].plot(episodes, sol_filtered_reward, color="red")
    fig.show()

    return ben_episode_reward, sol_episode_reward


def get_visited_nodes(list_solutions):
    vis_nodes = []
    for solution in list_solutions:
        for route in solution.routes:
            for edge in route.edges:
                if edge.end not in vis_nodes:
                    vis_nodes.append(edge.end)

    return vis_nodes


def extract_hist_data(nodes, seed):
    hist_data = []
    for node in nodes:
        weather = get_conditions(node.ID, seed)
        [wind_c, rain_c] = weather
        _, reward = getBinaryRandomReward(node_id=node.ID, wind=wind_c, rain=rain_c, seed=seed, verbose=False)
        hist_data.append([node.ID, wind_c, rain_c, reward])
        
    return hist_data


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

def compute_sol(fleetSize, routeMaxCost, nodes, sav_list, agent, hist_data, seed, alpha, alg = 0, verbose=False):
    """ Perform the BR edge-selection & routing-merging iterative process """
    # alg = 0 -> greedy; alg = 1 -> epsilon-greedy; alg = 2 -> UCB
    sol = dummySolution(routeMaxCost, nodes) # compute the dummy solution
    savList = copy.copy(sav_list) # make a shallow copy of the savings list since it will be modified
    # print(f"compute_sol seed: {seed}")

    # if isinstance(agent, list): # no RL agent, but historical data -> use savings
        # use historical records to recompute the efficiency based on average reward for each node
    if agent == None: # no RL agent, but historical data -> use savings
        savList = compute_avg_rewards(savList, hist_data, alpha)
        if verbose:
            print(hist_data)
            for node in nodes:
                print(f"{node}, {node.reward =}, {node.timesVisited}")
        use_agent = False
    else: # use RL agent
        use_agent = True
        if alg == 1:
            savList = compute_efficiency(savList, agent, seed, alpha)
        if alg == 2:
            savList = calculate_UCB(savList, agent, seed)
    
    progress_bar = tqdm(total=len(sav_list), desc="Processing", disable=not verbose)
    while len(savList) > 0: # list is not empty
        # The merging node is selected based on the reinf. learning algorithm
        # instead of a on biased-randomized position of an ordered efficiency list
        # recalculate effList
        if not use_agent or alg == 0: # greedy behaviour
            ijEdge = savList[0]
        else: # use RL agent
            if alg == 1:
                ijEdge = epsilonGreedy(savList)
            elif alg == 2:
                ijEdge = UCB(savList)
        if verbose:
            print(f"* Efficiency list lenght: {len(savList)} *")
            print(f"> merging {ijEdge}")
        ijEdge_idx = savList.index(ijEdge)
        savList.pop(ijEdge_idx)
        progress_bar.update(1)

        sol, mergeOk = merge_edge(sol, ijEdge, routeMaxCost)
        if mergeOk and verbose: print("> merge OK !!")
        if not mergeOk:
        # if still in list, delete edge (j, i) since it will not be used
            jiEdge = ijEdge.invEdge
            if jiEdge in savList:
                savList.remove(jiEdge)
                progress_bar.update(1)
        # printRoutes(sol)

    # sort the list of routes in sol by reward (reward) and delete extra routes
    sol.routes.sort(key = operator.attrgetter("reward"), reverse = True)
    for route in sol.routes[fleetSize:]:
        sol.reward -= route.reward # update reward
        sol.cost -= route.cost # update cost
        sol.routes.remove(route) # delete extra route

    progress_bar.close()
    return sol

def merge_edge(sol, ijEdge, routeMaxCost):
    # determine the nodes i < j that define the edge
    iNode = ijEdge.origin
    jNode = ijEdge.end
    # determine the routes associated with each node
    iRoute = iNode.inRoute
    jRoute = jNode.inRoute
    # check if merge is possible
    isMergeFeasible = checkMergingConditions(iNode, jNode, iRoute, jRoute, ijEdge, routeMaxCost, verbose=False)
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
    # else, merging is feasible
    return True

# def select_edge(savings_list, agent, historical_data, inst_seed, alg = 0):
#     """ Select edge to merge based on (alg=1) epsilon-greedy or (alg=2) Upper Confidence Bounds """
#     # TODO: integrate above to reduce calculation time
#     efficiency_list = savings_list
#     if alg == 0: # 0. Greedy (offline training)
#         edge = efficiency_list[0]
#     elif alg == 1: # 1. Epsilon-greedy
#         edge = epsilonGreedy(efficiency_list)
#     else: # 2. Upper Confidence Bounds
#         edge = UCB(efficiency_list, agent, historical_data, inst_seed)
#     return edge

def compute_efficiency(savings_list, rl_agent, inst_seed, alpha):
    eff_list = copy.copy(savings_list)
    for edge in eff_list:
        # determine the nodes i < j that define the edge
        iNode = edge.origin
        jNode = edge.end
        # determine the routes associated with each node
        # iRoute = iNode.inRoute
        # jRoute = jNode.inRoute
        # determine dynamic conditions for each node
        i_weather = get_conditions(iNode.ID, inst_seed)
        [wind_i, rain_i] = i_weather
        j_weather = get_conditions(jNode.ID, inst_seed)
        [wind_j, rain_j] = j_weather
        # print(f"compute_efficiency seed: {inst_seed}")
        #TODO: check that drone type is (1) the same in both routes, (2) allowed - number of drones is limited
        # iRoute, jRoute = assign_drones(iRoute, jRoute)
        # drone = iRoute.drone_type
        # determine rewards
        iNode.reward = rl_agent.get_estimated_reward(iNode, wind_i, rain_i)
        jNode.reward = rl_agent.get_estimated_reward(jNode, wind_j, rain_j)
        edgeReward = iNode.reward + jNode.reward
        # compute efficiency
        edge.efficiency = alpha * edge.savings + (1 - alpha) * edgeReward
    # sort the list of edges from higher to lower efficiency
    eff_list.sort(key = operator.attrgetter("efficiency"), reverse = True)

    return eff_list

def compute_avg_rewards(savings_list, historical_data, alpha):
    eff_list = copy.copy(savings_list)
    for event in historical_data:
        # print(f"{event =}")
        node_id = event[0]
        # print(f"{node_id =}")
        reward = event[-1]
        # print(f"{reward =}")
        node = find_node_in_eff_list(node_id, eff_list, verbose=False)
        # if find_node_in_eff_list(node_id, eff_list, verbose=True) is not None:
        # print(f"{node =}")
        if node is not None:
            if node.timesVisited == 0: # first time visit
                node.reward = reward
            else:
                # new_avg = old_avg+(reward-old_avg)/new_count
                node.reward = node.reward + (reward - node.reward)/node.timesVisited
                # node.reward += reward
            node.timesVisited += 1

    for edge in eff_list:
        # determine the nodes i < j that define the edge
        iNode = edge.origin
        jNode = edge.end
        edgeReward = iNode.reward + jNode.reward
        # compute efficiency
        edge.efficiency = alpha * edge.savings + (1 - alpha) * edgeReward
    # sort the list of edges from higher to lower efficiency
    eff_list.sort(key = operator.attrgetter("efficiency"), reverse = True)

    return eff_list

def find_node_in_eff_list(nodeID, efficiencyList, verbose=False):
    for edge in efficiencyList:
        node_ori = edge.origin
        node_end = edge.end
        # print(f"{node_ori.ID =}")
        if node_ori.ID == nodeID:
            return node_ori
        elif node_end.ID == nodeID:
            return node_end
    if verbose: print(f"{nodeID =} is not in efficiency list")
    return None

def epsilonGreedy(eff_list, epsilon = 0.1):
    random.seed(None) # initialize the random seed for the purpose of RL agent selection
    #TODO: extract epsilon as a global parameter
    epsilon = 0.1
    if random.random() < epsilon:
        # Explore: Select a random edge
        action = random.choice(eff_list)
    else:
        # Exploit: Select the edge with the highest efficiency
        action = eff_list[0]

    return action

def calculate_UCB(eff_list, rl_agent, inst_seed):
    """ Upper Confidence Bound algorithm """
    # count = 0
    edge_list = copy.copy(eff_list)
    # for edge in edge_list[:20]:
    for edge in edge_list:
        # if edge.UCB == float("inf"):
        iNode = edge.origin
        jNode = edge.end
        i_weather = get_conditions(iNode.ID, inst_seed)
        [wind_i, rain_i] = i_weather
        j_weather = get_conditions(jNode.ID, inst_seed)
        [wind_j, rain_j] = j_weather
        # determine rewards
        iValue = rl_agent.get_estimated_reward(iNode, wind_i, rain_i)
        jValue = rl_agent.get_estimated_reward(jNode, wind_j, rain_j)
        iTimes = rl_agent.get_times_visited(iNode, wind_i, rain_i)
        jTimes = rl_agent.get_times_visited(jNode, wind_j, rain_j)
        tTimes = rl_agent.get_total_visited()
        iNode_UCB = iValue + np.sqrt((2 * tTimes) / (iTimes + 1e-5))
        jNode_UCB = jValue + np.sqrt((2 * tTimes) / (jTimes + 1e-5))
        # compute upper confidence bound of the edge as the sum of the UCB of the nodes
        edge.UCB = iNode_UCB + jNode_UCB
        # count += 1
    # print(f"new iteration: count: {count}")
    # sort the list of edges from higher to lower efficiency and take the first one
    edge_list.sort(key = operator.attrgetter("UCB"), reverse = True)
    # action = edge_list[0]

    return edge_list

def UCB(edge_list):
    action = edge_list[0]

    return action


def run_test(file_name, train_episodes, alpha, testSeed, printFigures=True):
    # read instance data
    print("*** READING DATA ***")
    fleetSize, routeMaxCost, nodes = read_instance(file_name)

    # seeds: instance > test > episode
    instance_seed = file_name
    test_seed = instance_seed+testSeed #TODO: to be read from the test file: represents the variablity for that particular test

    # create RL agent
    agent = Agent(nodes)
    # print(agent.act_table)
    print("*** SIMULATING DATA ***")
    # use the file name as seed to ensure that all test are based on the same historical data
    historical_data = generate_historical_data(train_episodes, nodes, seed=file_name)
    # print(historical_data)
    print("*** TRAINING AGENT ***")
    train_agent(agent, historical_data)
    # print(agent.act_table)
    export_agent_to_file(agent, file_name)
    export_historical_data(historical_data, file_name)

    print("*** RUNNING HEURISTIC ***")
    savings_list = generateSavingsList(nodes)
    # print(savings_list)

    # weather
    # for node in nodes:
    #     print(f"{node}, {get_conditions(node, test_seed) =}")

    # compute benchmark
    print("*** COMPUTING BENCHMARK ***")
    benchmark = compute_sol(fleetSize, routeMaxCost, nodes, savings_list, None, historical_data, test_seed, alpha)   

    # compute solution
    print("*** COMPUTING SOLUTION ***")
    sol = compute_sol(fleetSize, routeMaxCost, nodes, savings_list, agent, historical_data, test_seed, alpha)   

    # emulate proposed solutions
    print("*** RESULTS BENCHMARK ***")
    # printRoutes(benchmark)
    emulation(benchmark, routeMaxCost, test_seed)
    # print(benchmark)

    print("*** RESULTS SOLUTION ***")
    # printRoutes(sol)
    emulation(sol, routeMaxCost, test_seed)
    # print(sol)
    
    if printFigures:
        figure1 = graphRoutes(nodes, benchmark, test_seed)
        figure2 = graphRoutes(nodes, sol, test_seed)
        figure1.show()
        figure2.show()
    
    # vis_nodes = get_visited_nodes([benchmark, sol])
    # print(vis_nodes)

    return benchmark.reward_sim, sol.reward_sim

def run_test_with_agent(file_name, agent, historical_data, alpha, testSeed, printFigures=True):
    # read instance data
    print("*** READING DATA ***")
    fleetSize, routeMaxCost, nodes = read_instance(file_name)

    # seeds: instance > test > episode
    instance_seed = file_name
    test_seed = instance_seed+testSeed #TODO: to be read from the test file: represents the variablity for that particular test

    print("*** RUNNING HEURISTIC ***")
    savings_list = generateSavingsList(nodes)

    # compute benchmark
    print("*** COMPUTING BENCHMARK ***")
    benchmark = compute_sol(fleetSize, routeMaxCost, nodes, savings_list, None, historical_data, test_seed, alpha)   

    # compute solution
    print("*** COMPUTING SOLUTION ***")
    sol = compute_sol(fleetSize, routeMaxCost, nodes, savings_list, agent, historical_data, test_seed, alpha)   

    # emulate proposed solutions
    print("*** RESULTS BENCHMARK ***")
    # printRoutes(benchmark)
    emulation(benchmark, routeMaxCost, test_seed)
    # print(benchmark)

    print("*** RESULTS SOLUTION ***")
    # printRoutes(sol)
    emulation(sol, routeMaxCost, test_seed)
    # print(sol)
    
    if printFigures:
        figure1 = graphRoutes(nodes, benchmark, test_seed)
        figure2 = graphRoutes(nodes, sol, test_seed)
        figure1.show()
        figure2.show()
    
    return benchmark.reward_sim, sol.reward_sim


if __name__ == "__main__":

    # file_name = r"data/p1.2.a.txt"
    # file_name = r"data/test_instance_03.txt"
    file_name = r"data/test_instance_v3_s1.txt"
    ALPHA = 0.01 # parameter to control the importance of expected rewards vs. savings in efficiency calculation
    TEST_SEED = str(2)

    # single run / offline-training
    # run_test(file_name, train_episodes=20, alpha=ALPHA, testSeed=TEST_SEED)

    # single run / offline-training - from trained agent
    # _, _, nodes = read_instance(file_name)
    # agent = Agent(nodes) # initialize agent
    # try:
    #     agent = import_agent_from_file(agent, file_name)
    #     historical_data = import_historical_data(file_name)
    #     run_test_with_agent(file_name, agent, historical_data, alpha=ALPHA, testSeed=TEST_SEED)
    # except:
    #     print("ERROR| no historical data or trained agent available")

    # multi-run / offline-training
    # _, _, nodes = read_instance(file_name)
    # agent = Agent(nodes) # initialize agent
    # for i in range(4):
    #     print(f"{i =}")
    #     try:
    #         agent = import_agent_from_file(agent, file_name)
    #         historical_data = import_historical_data(file_name)
    #         run_test_with_agent(file_name, agent, historical_data, alpha=ALPHA, testSeed=TEST_SEED+str(i), printFigures=True)
    #     except:
    #         run_test(file_name, train_episodes=10, alpha=ALPHA, testSeed=TEST_SEED+str(i), printFigures=True)

    # statistical significance test / offline-training
    # _, _, nodes = read_instance(file_name)
    # agent = Agent(nodes) # initialize agent
    # print(f"T-TEST for {file_name}")
    # ben_res = []
    # sol_res = []
    # for i in range(20):
    #     print(f"{i =}")
    #     try:
    #         agent = import_agent_from_file(agent, file_name)
    #         historical_data = import_historical_data(file_name)
    #         ben, sol = run_test_with_agent(file_name, agent, historical_data, alpha=ALPHA, testSeed=TEST_SEED+str(i), printFigures=False)
    #     except:
    #         ben, sol = run_test(file_name, train_episodes=10, alpha=ALPHA, testSeed=TEST_SEED+str(i), printFigures=False)
    #     ben_res.append(ben)
    #     sol_res.append(sol)
    # print(f"{ben_res =}")
    # print(f"{sol_res =}")
    # ttest = stats.ttest_ind(ben_res, sol_res)
    # print(ttest)

    # multistart / online-training
    # alg = 0 -> greedy; alg = 1 -> epsilon-greedy; alg = 2 -> UCB
    benchmark, sol = multistart(file_name, n_episodes=50, alpha=ALPHA, alg=1, testSeed=TEST_SEED)
    print(benchmark)
    print(sol)

