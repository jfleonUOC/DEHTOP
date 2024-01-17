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
    for node in nodes:
        if get_conditions(node, seed) == 1:
            x_1.append(node.x)
            y_1.append(node.y)
        else:
            x_0.append(node.x)
            y_0.append(node.y)
    plt.plot(x_1, y_1, "^")
    plt.plot(x_0, y_0, "v")
    
    # plt.show()

    return fig 
