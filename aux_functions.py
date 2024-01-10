''' Reviewed by Angel A. Juan 2023.06 for the TOP with stochastic / dynamic travel times '''

from aux_objects import Test, Node

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
