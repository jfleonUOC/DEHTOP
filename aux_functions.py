import random
import math
import matplotlib.pyplot as plt
from aux_objects import Test, Node
from emulation import get_conditions

def read_tests(file_name):
    """ Generate a list of tests to run from a file """
    with open(file_name) as file:
        tests = []
        for line in file:
            tokens = line.split("\t")
            if '#' not in tokens[0]:
                aTest = Test(*tokens) # '*' unpacks tokens as parameters
                tests.append(aTest)
    return tests

def read_instance(file_name):
    """ Generate a list of nodes from instance file """
    with open(file_name) as instance:
        i = -3 # we start at -3 so that the first node is node 0
        nodes = []
        for line in instance:
            if i == -3: pass # line 0 contains the number of nodes, not needed
            elif i == -2: fleetSize = int(line.split(';')[1])
            elif i == -1: routeMaxCost = float(line.split(';')[1])
            else:
                # array data with node data: x, y, reward
                data = [float(x) for x in line.split(';')]
                # create instance nodes
                aNode = Node(i, data[0], data[1], data[2])
                nodes.append(aNode)
            i += 1
    return fleetSize, routeMaxCost, nodes

def printRoutes(sol):
    """ Print routes in a solution """
    for route in sol.routes:
        print("0", end = "")
        for e in route.edges:
            print("->", e.end.ID, end="")
        print("\nRoute det reward:", route.reward, "; det cost:", route.cost)


def graphRoutes_old01(nodes, sol, seed):
    # unknown nodes in solution are shown as red
    fig = plt.figure()
    for route in sol.routes:
        x = []
        y = []
        # start node
        start_node = route.edges[0].origin
        x.append(start_node.x)
        y.append(start_node.y)
        # loop nodes
        for edge in route.edges:
            node = edge.end
            x.append(node.x)
            y.append(node.y)

        # plot
        plt.plot(x, y, "-o")

        # anotate
        # start node
        plt.text(x[0],y[0],"start")
        # loop nodes
        for edge in route.edges[:-1]:
            node = edge.end
            if node.realReward == 1:
                symbol = "*"
            else:
                symbol = ""
            plt.text(node.x, node.y, symbol)
        # last node
        plt.text(x[-1], y[-1], "end")

    # add nodes with conditions
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    for node in nodes:
        # if get_conditions(node, seed) == 1:
        print(f"emulating node {node}, prob: {node.probability}")
        if node.probability > 0.5:
            x_1.append(node.x)
            y_1.append(node.y)
        else:
            x_0.append(node.x)
            y_0.append(node.y)
    plt.plot(x_1, y_1, marker="^", color="g", ms=8, linestyle="None")
    plt.plot(x_0, y_0, marker="v", color="r", ms=8, linestyle="None")
    
    # plt.show()

    return fig 

def graphRoutes(nodes, sol, seed):
    fig = plt.figure()
    for route in sol.routes:
        x = []
        y = []
        # start node
        start_node = route.edges[0].origin
        x.append(start_node.x)
        y.append(start_node.y)
        # loop nodes
        for edge in route.edges:
            node = edge.end
            x.append(node.x)
            y.append(node.y)

        # plot
        plt.plot(x, y, "-o")

        # anotate
        # start node
        plt.text(x[0],y[0],"start")
        # loop nodes
        for edge in route.edges[:-1]:
            node = edge.end
            if node.realReward == 1:
                symbol = "*"
            else:
                symbol = ""
            plt.text(node.x, node.y, symbol)
        # last node
        plt.text(x[-1], y[-1], "end")

    # add nodes with conditions
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    x_n = []
    y_n = []
    for node in nodes:
        # if get_conditions(node, seed) == 1:
        # print(f"emulating node {node}, prob: {node.probability}")
        if node.probability is not None:
            if node.probability > 0.5:
                x_1.append(node.x)
                y_1.append(node.y)
            else:
                x_0.append(node.x)
                y_0.append(node.y)
        else:
            x_n.append(node.x)
            y_n.append(node.y)
    plt.plot(x_1, y_1, marker="^", color="g", ms=8, linestyle="None")
    plt.plot(x_0, y_0, marker="v", color="r", ms=8, linestyle="None")
    plt.plot(x_n, y_n, marker="o", color="gray", ms=8, linestyle="None")
    
    # plt.show()

    return fig 

def generateInstance_old1(save_path, n_nodes, n_veh, max_cost, spread=20, seed=0):
    rows = []
    # config rows
    first_row = f"n;{n_nodes}\n"
    second_row = f"n;{n_veh}\n"
    third_row = f"tmax;{max_cost}\n"
    rows.extend([first_row, second_row, third_row])

    # add nodes
    random.seed(seed)
    for node in range(n_nodes):
        x = random.uniform(0, spread)
        y = random.uniform(0, spread)
        coords = f"{x:.3f};{y:.3f};0\n"
        rows.append(coords)

    # order first and final
    print(rows)

    # write instance
    with open(save_path, "w") as outfile:
        # outfile.write()
        outfile.writelines(rows)
    print(f"New instance created at: {save_path}")


def generateInstance_old2(save_path, n_nodes, n_veh, spread, seed=0):
    rows = []
    max_cost = spread * 1.8
    # config rows
    first_row = f"n;{n_nodes}\n"
    second_row = f"n;{n_veh}\n"
    third_row = f"tmax;{max_cost}\n"
    rows.extend([first_row, second_row, third_row])

    # add nodes
    random.seed(seed)
    for node in range(n_nodes-2):
        x = random.uniform(0, spread)
        y = random.uniform(0, spread)
        coords = f"{x:.3f};{y:.3f};0\n"
        rows.append(coords)

    # add first and final
    first_node = f"{0:.3f};{spread/2:.3f};0\n" 
    last_node = f"{spread:.3f};{spread/2:.3f};0" 
    rows.insert(3,first_node)
    rows.append(last_node)
    print(rows)

    # write instance
    with open(save_path, "w") as outfile:
        # outfile.write()
        outfile.writelines(rows)
    print(f"New instance created at: {save_path}")

def generateInstance(save_path, n_veh):
    rows = []
    DENS = 0.5 # density: number of point per squared unit
    n_nodes = n_veh * 20 # if this is modify, check the position of nodes!!
    size = round(math.sqrt(n_nodes / DENS))
    max_cost = size * 3
    # config rows
    first_row = f"n;{n_nodes}\n"
    second_row = f"n;{n_veh}\n"
    third_row = f"tmax;{max_cost}\n"
    rows.extend([first_row, second_row, third_row])

    # add nodes
    # randomly
    # random.seed(seed)
    # for node in range(n_nodes-2):
    #     x = random.uniform(0, size)
    #     y = random.uniform(0, size)
    #     coords = f"{x:.3f};{y:.3f};0\n"
    #     rows.append(coords)
    # uniformly in a square
    Nx = math.ceil(math.sqrt(n_nodes))
    Ny = math.floor(math.sqrt(n_nodes))
    for p_x in range(Nx):
        x = (1/(2*(size/Nx))) + p_x*(size/Nx)
        for p_y in range(Ny):
            y = (1/(2*(size/Ny))) + p_y*(size/Ny)
            coords = f"{x:.3f};{y:.3f};0\n"
            rows.append(coords)

    # add first and final
    first_node = f"{0:.3f};{size/2:.3f};0\n" 
    last_node = f"{size:.3f};{size/2:.3f};0" 
    rows.insert(3,first_node)
    rows.append(last_node)
    print(rows)

    # write instance
    with open(save_path, "w") as outfile:
        # outfile.write()
        outfile.writelines(rows)
    print(f"New instance created at: {save_path}")

if __name__ == "__main__":
    n_veh = 3
    seed = 1
    file_name = f"data/test_instance_v{n_veh}_s{seed}.txt"
    generateInstance(file_name, n_veh=n_veh)