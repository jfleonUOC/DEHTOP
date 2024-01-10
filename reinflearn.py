import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from emulation import getBinaryRandomReward
from aux_objects import Node

class Agent:
    def __init__(self, nodes, drones = ["A","W"], weather = [0,1], epsilon=0.1):
        self.nodes = nodes # List of objects of class Node
        self.drones = drones
        self.weather = weather
        self.epsilon = epsilon

        # Initialize action-table with zeros
        self.act_table = self.initialize_act_table()

    def initialize_act_table(self):
        # node_list = [x for x in range(1, nodes+1)]
        node_list = [node.ID for node in self.nodes]
        combinations = list(product(self.drones, self.weather, node_list, [0], [0]))
        columns_names = ["drones","weather","nodes", "value", "visited"]
        table = pd.DataFrame(combinations, columns=columns_names)

        return table

    def select_action(self, drone_type, weather_type):
        filtered_table = self.act_table.loc[(self.act_table["drones"] == drone_type) & (self.act_table["weather"] == weather_type)]
        # Exploration-exploitation strategy (epsilon-greedy)
        if np.random.rand() < self.epsilon:
            # Explore: Select a random action
            row_index = filtered_table.sample().index.item()
            action = self.act_table.at[row_index, "nodes"]
        else:
            # Exploit: Select the action with the highest action-value
            action = filtered_table.loc[filtered_table["value"].idxmax()]["nodes"]

        return action

    def update_act_value(self, drone_type, weather_type, node, reward):
        # action-value update using the sample-average method
        row_index = self.act_table.loc[(self.act_table["drones"] == drone_type) & (self.act_table["weather"] == weather_type) & (self.act_table["nodes"] == node)].index.item()
        # self.act_table.at[row_index,"visited"] += 1
        # self.act_table.at[row_index,"value"] = (self.act_table.at[row_index,"value"] + reward) / self.act_table.at[row_index,"visited"]
        old_avg = self.act_table.at[row_index,"value"] 
        old_count = self.act_table.at[row_index,"visited"] 
        new_count = old_count + 1
        new_avg = old_avg+(reward-old_avg)/new_count
        self.act_table.at[row_index,"value"] = new_avg
        self.act_table.at[row_index,"visited"] = new_count


    def get_best_action(self, drone_type, weather_type):
        # Retrieve the best action based on the highest action-value
        filtered_table = self.act_table.loc[(self.act_table["drones"] == drone_type) & (self.act_table["weather"] == weather_type)]
        best_action = filtered_table.loc[filtered_table["value"].idxmax()]["nodes"]

        return best_action

    def get_estimated_reward(self, node, drone_type, weather_type):
        # Extract from Q-table the estimated reward for a given {node, drone, weather}
        reward = self.act_table.loc[(self.act_table["nodes"]==node.ID)&(self.act_table["drones"]==drone_type)&(self.act_table["weather"]==weather_type)]["value"].item()

        return reward

def generate_historical_data(events, nodes, seed, drones = ["A","W"], weather = [0,1]):
    """ Generate historical data to train the agent """
    historical_data = []
    # node_list = [x for x in range(1, nodes+1)]
    node_list = [node.ID for node in nodes]
    combinations = list(product(drones, weather, node_list))
    for _ in tqdm(range(events)):
        for comb in combinations:
            reward = getBinaryRandomReward(*comb, seed, verbose=False)
            event = list(comb)
            event.append(reward)
            historical_data.append(event)

    return historical_data

def train_agent(agent, historical_data):
    """ Train the agent based on historical data """
    for event in tqdm(historical_data):
        drone, weather, node, reward = event
        agent.update_act_value(drone, weather, node, reward)
    return

if __name__ == "__main__":
    # create dummy nodes
    nodes = []
    for i in range(4):
        nodes.append(Node(i,0,0,0))

    agent = Agent(nodes) # Create an agent
    # agent.update_act_value("A",1,3,10)
    # agent.update_act_value("A",1,3,20)
    # agent.update_act_value("A",1,4,20)
    print(agent.act_table)


    historical_data = generate_historical_data(100, nodes, 0)
    # print(historical_data)
    train_agent(agent, historical_data)
    print(agent.act_table)

    print(agent.select_action("A",1))
    print(agent.get_best_action("A",1))
    print(agent.get_estimated_reward(nodes[0],"A",1))
