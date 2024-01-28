# import numpy as np
import pickle
import pandas as pd
from itertools import product
from tqdm import tqdm
from emulation import getBinaryRandomReward
from aux_objects import Node

class Agent:
    def __init__(self, nodes, weather = []):
        self.nodes = nodes # List of objects of class Node
        self.weather = weather

        # weather
        wind = rain = visi = temp = [-2, -1, 0, 1, 2]
        self.weather.extend([wind, rain, visi, temp])

        # Initialize action-value table with zeros
        self.act_table = self.initialize_act_table()

    def initialize_act_table(self):
        """ Initialize the action-value table with zeros """
        node_list = [node.ID for node in self.nodes]
        combinations = list(product(node_list, *self.weather, [0], [0]))
        columns_names = ["nodes", "wind", "rain", "visi", "temp", "value", "visited"]
        table = pd.DataFrame(combinations, columns=columns_names)

        return table

    def update_act_value(self, node, wind, rain, visi, temp, reward):
        """ action-value update using the sample-average method """
        # columns_names = ["nodes", "wind", "rain", "visi", "temp", "value", "visited"]
        row_index = self.act_table.loc[(self.act_table["nodes"] == node)
                                       & (self.act_table["wind"] == wind)
                                       & (self.act_table["rain"] == rain)
                                       & (self.act_table["visi"] == visi)
                                       & (self.act_table["temp"] == temp)
                                       ].index.item()
        old_avg = self.act_table.at[row_index,"value"] 
        old_count = self.act_table.at[row_index,"visited"] 
        new_count = old_count + 1
        new_avg = old_avg+(reward-old_avg)/new_count
        self.act_table.at[row_index,"value"] = new_avg
        self.act_table.at[row_index,"visited"] = new_count

    def get_estimated_reward(self, node, wind, rain, visi, temp):
        """ Extract from Q-table the estimated reward for a given node and set of weather conditions """
        # columns_names = ["nodes", "wind", "rain", "visi", "temp", "value", "visited"]
        reward = self.act_table.loc[(self.act_table["nodes"] == node.ID)
                                    & (self.act_table["wind"] == wind)
                                    & (self.act_table["rain"] == rain)
                                    & (self.act_table["visi"] == visi)
                                    & (self.act_table["temp"] == temp)
                                    ]["value"].item()

        return reward

def generate_historical_data(n_events, nodes, seed):
    """ Generate historical data to train the agent """
    # simulate every combination n_event number of times
    historical_data = []
    node_list = [node.ID for node in nodes]
    wind = rain = visi = temp = [-2, -1, 0, 1, 2]
    combinations = list(product(node_list, wind, rain, visi, temp))
    # print(combinations)
    for i in tqdm(range(n_events)):
        for comb in combinations:
            _, reward = getBinaryRandomReward(*comb, seed+str(i), verbose=False)
            event = list(comb)
            event.append(reward)
            historical_data.append(event)

    return historical_data

def train_agent(agent, historical_data, verbose=True):
    """ Train the agent based on historical data """
    for event in tqdm(historical_data, disable=not verbose):
        # columns_names = ["nodes", "wind", "rain", "visi", "temp", "value", "visited"]
        node, wind, rain, visi, temp, reward = event
        agent.update_act_value(node, wind, rain, visi, temp, reward)
    return

def export_agent_to_file(agent, output, verbose=True):
    file_output = output[:-4] + "_agent.csv"
    agent.act_table.to_csv(file_output)
    if verbose: print(f"> agent exported to {file_output} ({agent.act_table.shape[0]} lines)")
    return

def import_agent_from_file(agent, input, verbose=True):
    file_input = input[:-4] + "_agent.csv"
    df_input = pd.read_csv(file_input)
    len_input = df_input.shape[0]
    len_base = agent.act_table.shape[0] 
    if len_input == len_base:
        agent.act_table = df_input
        if verbose: print(f"> agent imported from {file_input} ({agent.act_table.shape[0]} lines)")
    else:
        print(f"ERROR| cannot import agent - lenght is different ({len_input} vs. {len_base})")

    return agent

def export_historical_data(historical_data, output):
    file_output = output[:-4] + "_histdata"
    with open(file_output, "wb") as file:
        pickle.dump(historical_data, file)

    return

def import_historical_data(input):
    file_input = input[:-4] + "_histdata"
    with open(file_input, "rb") as file:
        historical_data = pickle.load(file)

    return historical_data


if __name__ == "__main__":
    # create dummy nodes
    nodes = []
    for i in range(4):
        nodes.append(Node(ID=i, x=0, y=0, reward=0))

    agent = Agent(nodes) # Create an agent
    print(agent.act_table)

    seed = "test_instance.txt4827"
    historical_data = generate_historical_data(10, nodes, seed)
    # print(historical_data)
    train_agent(agent, historical_data)
    print(agent.act_table)
    print(agent.get_estimated_reward(node=nodes[0], wind=-2, rain=-2, visi=-2, temp=-2))
