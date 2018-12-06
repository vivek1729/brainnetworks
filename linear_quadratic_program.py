import pulp
import csv
import numpy
import time
from pulp import *
from cvxopt import matrix, solvers

'''
Get adj_matrix from txt files.

Inputs:
filename - filename of a comma separated value file

Output:
adj_matrix - n*n matrix for n vertices where adj_matrix[i][j] represents a directed edge from i to j if its 1
'''
def read_txt(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        adj_matrix = list(reader)
    adj_matrix = [[int(j) for j in i[0:(len(i))]] for i in adj_matrix]
#     sinks = [any(row) for row in adj_matrix]
#     sinks = [i for i, v in enumerate(sinks) if not v]
#     adj_matrix_t = list(map(list, zip(*adj_matrix)))
#     sources = [any(row) for row in adj_matrix_t]
#     sources = [i for i, v in enumerate(sources) if not v]
    return adj_matrix

def write_txt(filename, Matrix):
#     Matrix = read_txt('test.csv')
    with open(filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        [spamwriter.writerow(row) for row in Matrix]

def nrow(Matrix):
    return(len(Matrix))

def ncol(Matrix):
    return(len(Matrix[0]))


'''
Helper utility to preprocess our input graph

Inputs:
adj_matrix - n*n matrix for n vertices where adj_matrix[i][j] represents a directed edge from i to j if its 1
sources - list of vertices that are sources eg: [0,2] where 0,2 would be valid rows and columns in adj_matrix
sinks - list of vertices that are sinks eg: [3]

Ouputs:
adj_matrix - A new adj_matrix where a super source and super sink are added to handle multiple source/sink case
edge_cap_facs - A list of variables representing edge capacity scaling factors (Our decision variable x)
                Values would be of the form x_i_j which denotes capacity scaling factor for edge going from vertex i to j
edge_flow_vals - A list of variables representing edge flow values (Variable y in LP)
                 Values would be of the form y_i_j which denotes flow for edge going from vertex i to j
vertex_dict - A list of dictionaries of form [{i:[],o:[]},{i:[],o:[]},..] 
              where i and o are input and output edges for that vertex (represented by index in this list)
              the edges here would just be the corresponding edge flow variables defined in the edge_flow_vals list
              so that this can be directly used to create flow conservation constraints in the LP
'''
def preprocess_graph(adj_matrix, sources, sinks):
    n = len(adj_matrix) #Infer the size of adj_matrix
    m = n+2 #New size of adjacency matrix as we add 2 new nodes, 1 super source and 1 super sink
    
    #Initialize the new_adj_matrix with zeros
    new_adj_matrix = []
    for i in range(m):
        new_adj_matrix.append([])
        for j in range(m):
            new_adj_matrix[i].append(0)
    #Copy over values from the original adj_matrix, 
    #basically the ith vertex in original matrix would become the (i+1)th vertex in the new matrix
    for i in range(n):
        new_adj_matrix[i+1][1:m-1] = adj_matrix[i]
    
    #Connect sources and sinks to new super source and super sink
    #0 would always be the super source and m would always be the super sink
    for s in sources:
        new_adj_matrix[0][s+1] = 1

    for t in sinks:
        new_adj_matrix[t+1][m-1] = 1
        
    #The new adjacency matrix is created. 
    #Now we need to create the edge capacity factors, edge flow factors and vertex dict
    edge_capacity_factors = []
    edge_flows = []
    vertex_dict = []
    #Initialize vertex dict
    for i in range(len(new_adj_matrix)):
        vertex_dict.append({'i':[],'o':[]})
        
    for i in range(len(new_adj_matrix)):
        for j in range(len(new_adj_matrix[i])):
            if(new_adj_matrix[i][j] == 1):
                ecf = 'x_'+str(i)+'_'+str(j)
                ef = 'y_'+str(i)+'_'+str(j)
                edge_capacity_factors.append(ecf)
                edge_flows.append(ef)
                #Outgoing for i
                vertex_dict[i]['o'].append(ef)
                #Incoming for j
                vertex_dict[j]['i'].append(ef)
                
    return new_adj_matrix, edge_capacity_factors, edge_flows, vertex_dict

'''
Cost function

Inputs:
x       - capacity of edge
edge    - edge indice

Outputs:
y       - cost of assigning capacity of an edge to x
'''
def cost_function(x, edge):
    return x-1

def cost_function2(x, edge):
    return (x-1)*(x-1)

'''
calculate cost using LP Solver

Inputs:
adj_matrix    - n*n matrix represented as a list of lists in python as shown below, 1 represents an edge from i->j
sources       - list of vertices that are sources
sinks         - list of vertices that are sinks
flow_val      - Value of flow that we want to reinstate
cost_function - cost function 
'''
def process_graph_LPSolver(adj_matrix, sources, sinks, flow_val, cost_function):
    #Preprocess graph, prepare variables and vertex dictionary for LP program
    transition_matrix,edge_capacity_factors,edge_flows,vertex_dict = preprocess_graph(adj_matrix,sources,sinks)
    source = 0 #super source
    sink = len(transition_matrix)-1 #super sink
    CAP_UB = flow_val * len(adj_matrix)
    cap_dict = {}
    for i in range(len(transition_matrix[source])):
        if(transition_matrix[0][i] == 1):
            cf = 'x_0'+'_'+str(i)
            cap_dict[cf] = len(vertex_dict[i]['o'])

    for i in range(len(sinks)):
        new_sink_idx = sinks[i]+1
        cf = 'x_'+str(new_sink_idx)+'_'+str(sink)
        cap_dict[cf] = len(vertex_dict[new_sink_idx]['i'])
    
    '''
    LP Problem formulation
    '''
    lp_problem = pulp.LpProblem("Graph min cost", pulp.LpMinimize)
    edge_cap_vars = LpVariable.dicts("EdgeCaps",edge_capacity_factors,1)
    edge_flow_vars = LpVariable.dicts("EdgeFlows",edge_flows,0)

    #Objective function
    lp_problem += lpSum([(edge_cap_vars[i] - 1) for i in edge_capacity_factors]), "Z"

    #Flow conservation constraints for nodes besides source and sink
    for i in range(len(vertex_dict)):
        if(i != source and i != sink):
            lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[i]['i']]) >= lpSum([edge_flow_vars[k] for k in vertex_dict[i]['o']])
            lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[i]['i']]) <= lpSum([edge_flow_vars[k] for k in vertex_dict[i]['o']])

    #Flow conservation for source
    lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[source]['o']]) >= flow_val
    lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[source]['o']]) <= flow_val

    #Flow conservation for sink
    lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[sink]['i']]) >= flow_val
    lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[sink]['i']]) <= flow_val

    #Capacity constraints
    for cap, flow in zip(edge_capacity_factors, edge_flows):
        cap_multiplier = 1
        if(cap in cap_dict):
            cap_multiplier = CAP_UB
        lp_problem += edge_flow_vars[flow] <= edge_cap_vars[cap]*cap_multiplier
    
    #Solve LP_problem
    lp_problem.solve()
    
    assert(pulp.LpStatus[lp_problem.status] == 'Optimal')
    
    #compute total cost
    capValues = [v.varValue for v in lp_problem.variables()[0:(len(edge_cap_vars))]]
    cost = sum([cost_function(cap, 0) for cap in capValues])
    
    return cost

def process_graph_QPSolver(adj_matrix, sources, sinks, flow_val, cost_function):
    #Preprocess graph, prepare variables and vertex dictionary for LP program
    transition_matrix,edge_capacity_factors,edge_flows,vertex_dict = preprocess_graph(adj_matrix,sources,sinks)
    source = 0 #super source
    sink = len(transition_matrix)-1 #super sink
    CAP_UB = 100#flow_val * len(adj_matrix)
    cap_dict = {}
    for i in range(len(transition_matrix[source])):
        if(transition_matrix[0][i] == 1):
            cf = 'x_0'+'_'+str(i)
            cap_dict[cf] = len(vertex_dict[i]['o'])

    for i in range(len(sinks)):
        new_sink_idx = sinks[i]+1
        cf = 'x_'+str(new_sink_idx)+'_'+str(sink)
        cap_dict[cf] = len(vertex_dict[new_sink_idx]['i'])

    # Need to make matrices P,q,G,h
    # minimizing 1/2 * X_transpose * P * X + q_transpose * X
    # subject to G*X <= h

    '''
    Matrix P. number of rows/columns is equal to the total number of capacity and flow variables
    Objective function uses P and q matrices, which will have nonzero values for capacity
    (since only they appear in the objective function)
    '''

    p_diag1 = [2] * len(edge_capacity_factors)
    p_diag2 = [0] * len(edge_flows)
    p_diag = p_diag1 + p_diag2
    P = matrix(numpy.diag(p_diag), tc='d')
    
    '''q is just a 1-d matrix
    should have the same coeffs as p_diag (except negative), so just use that'''
    q = matrix(numpy.array(p_diag)*-1, tc='d')
    
    '''
    Gx <= h (rest of the constraints)
    '''
    G_array = []
    h_array = []

    edge_flows_start_index = len(edge_capacity_factors)

    # lets do capacity constraints first
    for cap, flow in zip(edge_capacity_factors, edge_flows):
        cap_multiplier = 1
        if(cap in cap_dict):
            cap_multiplier = CAP_UB

        temp_array = [0] * (len(edge_capacity_factors) + len(edge_flows))
        cap_index = edge_capacity_factors.index(cap)
        temp_array[cap_index] = -1 * cap_multiplier

        flow_index = edge_flows_start_index + edge_flows.index(flow)
        temp_array[flow_index] = 1
        #print(temp_array[:len(temp_array)//2])
        #print(temp_array[len(temp_array)//2:])
        #input()
        G_array.append(temp_array)
        h_array += [0]
    
    '''
    lets do flow conservation for everything but source/sinks
    no >= operator, so convert to <=
    lpSum([edge_flow_vars[j] for j in vertex_dict[i]['i']]) >= lpSum([edge_flow_vars[k] for k in vertex_dict[i]['o']])
    lpSum([edge_flow_vars[j] for j in vertex_dict[i]['i']]) - lpSum([edge_flow_vars[k] for k in vertex_dict[i]['o']]) >= 0
    - lpSum([edge_flow_vars[j] for j in vertex_dict[i]['i']]) + lpSum([edge_flow_vars[k] for k in vertex_dict[i]['o']]) <= 0

    '''
    for i in range(len(vertex_dict)):
        if(i != source and i != sink):
            temp_array = [0] * (len(edge_capacity_factors) + len(edge_flows))

            for j in vertex_dict[i]['i']:
                temp_array[edge_flows_start_index + edge_flows.index(j)] = -1

            for j in vertex_dict[i]['o']:
                temp_array[edge_flows_start_index + edge_flows.index(j)] = 1

            #t1 = temp_array[:len(temp_array)//2]
            #t2 = temp_array[len(temp_array)//2:]
            #print (t1)
            #print (t2)
            #input()

            G_array.append(temp_array)
            h_array += [0]

    '''
    Now flow conservation for source

    lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[source]['o']]) >= flow_val
    lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[source]['o']]) <= flow_val
    '''

    temp_array = [0] * (len(edge_capacity_factors) + len(edge_flows))
    for j in vertex_dict[source]['o']:
        temp_array[edge_flows_start_index + edge_flows.index(j)] = -1
    G_array.append(temp_array)
    h_array += [-1 * flow_val]

    

    temp_array = [0] * (len(edge_capacity_factors) + len(edge_flows))
    for j in vertex_dict[source]['o']:
        temp_array[edge_flows_start_index + edge_flows.index(j)] = 1
    G_array.append(temp_array)
    h_array += [flow_val]

    

    '''
    Now flow conservation for sink

    lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[sink]['i']]) >= flow_val
    lp_problem += lpSum([edge_flow_vars[j] for j in vertex_dict[sink]['i']]) <= flow_val
    '''
    temp_array = [0] * (len(edge_capacity_factors) + len(edge_flows))
    for j in vertex_dict[sink]['i']:
        temp_array[edge_flows_start_index + edge_flows.index(j)] = -1
    G_array.append(temp_array)
    h_array += [-1 * flow_val]

    temp_array = [0] * (len(edge_capacity_factors) + len(edge_flows))
    for j in vertex_dict[sink]['i']:
        temp_array[edge_flows_start_index + edge_flows.index(j)] = 1
    G_array.append(temp_array)
    h_array += [flow_val]

    # make sure cap,flow vals >= 0
    # so we add 
    #   - flow <= 0
    #   - cap <= 0
    for i in range(0,(len(edge_capacity_factors) + len(edge_flows))):
        temp_array = [0] * (len(edge_capacity_factors) + len(edge_flows))
        temp_array[i] = -1
        #print(temp_array)
        #input()
        G_array.append(temp_array)
        if(i >= len(edge_capacity_factors)):
            h_array += [1]
        else:
            h_array += [0]

    #  solve it!
    G = matrix(numpy.array(G_array),tc='d')    
    h = matrix(numpy.array(h_array),tc='d')
    #solvers.options['refinement']=2
    sol = solvers.qp(P,q,G,h)
    #print (sol['status'])

    optimal_values = sol['x']
    #print (optimal_values)

    optimal_values = optimal_values[:len(optimal_values)//2]
    flow_values = optimal_values[len(optimal_values)//2:]
    k = 0
    #for i,j in zip(optimal_values,flow_values):
    #    if(i < j):
    #        if(not k in sources):
    #            print ("BAD!!!")
    #    k += 1

    #print (optimal_values)
    cost = sum([cost_function(val, 0) for val in optimal_values])
    return cost

def test_linear_bnds(pathIn, pathOut):
    with open(pathIn + 'allComb.csv', 'r') as f:
        reader = csv.reader(f)
        allComb = list(reader)
    for iRow in list(range(0, nrow(allComb))):
        print("processing: "+str(iRow))
        print(str(allComb[iRow]))
        adj_matA = read_txt(filename=pathIn + str(iRow+1) + 'MBefore.csv')
        adj_matB = read_txt(filename=pathIn + str(iRow+1) + 'MAfter.csv')
        nSize   = nrow(adj_matA)
        nSource = int(allComb[iRow][0])
        nSink   = int(allComb[iRow][0])
        nTrauma = int(allComb[iRow][1])
        maxflowBefore = int(allComb[iRow][2])
        maxflowAfter  = int(allComb[iRow][3])
        
        timeStart = time.time()
        costAfter   = process_graph_QPSolver(adj_matrix=adj_matB, 
                                              cost_function=cost_function2, 
                                              flow_val=maxflowBefore, 
                                              sources= list(range(0, nSource)),
                                              sinks = list(range(nSize-nSink, nSize)))
        timeCost  = time.time() - timeStart
        allComb[iRow].append(costAfter)
        allComb[iRow].append(timeCost)
        print(str(allComb[iRow]))
    write_txt(pathOut + 'allComb_out.csv', allComb)

def main():
    # adj_matrix = [[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
    # [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], 
    # [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 
    # [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], 
    # [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0], 
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    '''test for process_graph_LPSolver'''
    #     adj_matrix = read_txt('Adj_Matrix0.txt')
    #     sources = [0,1,2]
    #     sinks = [9,10,11]
    #     flow_val = 3
    #     cost = process_graph_LPSolver(adj_matrix, sources, sinks, flow_val, cost_function)
    #     print(cost)
    '''test for linear cost function'''
#     pathIn  = 'testcases/linear/';
#     pathOut = 'testcases/linear/'
#     test_linear(pathIn, pathOut)
#     pathIn  = 'testcases/sizeofgraphset2/';
#     pathOut = 'testcases/sizeofgraphset2/'
#     test_linear(pathIn, pathOut)
    '''testcase from bn_datasets'''
    pathIn   = 'bn_dataset/out/bn-cat-mixed-species_brain_1.txt';
    pathOut  = 'bn_dataset/out/bn-cat-mixed-species_brain_1.txt';
    test_linear_bnds(pathIn, pathOut)
    pathIn   = 'bn_dataset/out/bn-fly-drosophila_medulla_1.txt';
    pathOut  = 'bn_dataset/out/bn-fly-drosophila_medulla_1.txt';
    test_linear_bnds(pathIn, pathOut)
    pathIn   = 'bn_dataset/out/bn-macaque-rhesus_brain_1.txt';
    pathOut  = 'bn_dataset/out/bn-macaque-rhesus_brain_1.txt';
    test_linear_bnds(pathIn, pathOut)
    pathIn   = 'bn_dataset/out/bn-mouse_brain_1.txt';
    pathOut  = 'bn_dataset/out/bn-mouse_brain_1.txt';
    test_linear_bnds(pathIn, pathOut)

'''
adj_matrix =[[0,1,1,0],
    [0,0,0,1],
    [0,0,0,1],
    [0,0,0,0]]
sources = [0]
sinks = [3]
flow_val = 3
cost = process_graph_QPSolver(adj_matrix, sources, sinks, flow_val, cost_function2)

print (cost)
'''

main()
