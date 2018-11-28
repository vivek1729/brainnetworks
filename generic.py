import pulp as _pulp
import numpy as _np

'''
Graph description

v0 -> v2
v0 -> v1
v0 -> v3

v1 -> v3
v2 -> v3


   v0
  / |\ 
 /  | \
v1  |  v2
 \  | /
  \ |/ 
   v3

All the initial capacities are 1. Max flow is 3

Cost function
Cost of increasing capacity on any edge by factor of x is x.


-----------------------------------

For our purposes, assume we have some transition matrix for this graph
This will be 4x4 matrix, which looks like the following:
[0 1 1 1
 0 0 0 1
 0 0 0 1
 0 0 0 0]

This next part is simply to give an example.
In the future, we assume we're given this matrix

Note that, since initial capacities are all 1, the adjacency/transition matrix encodes this automatically
if there is an edge
transition_matrix is an np.arrays
'''
transition_matrix = _np.zeros((4,4))
transition_matrix[0,1] = 1
transition_matrix[0,2] = 1
transition_matrix[0,3] = 1
transition_matrix[1,3] = 1
transition_matrix[2,3] = 1
dimension = transition_matrix.shape[0] # in this case, (4,4) = 4


'''We will simulate trauma by removing the edge from v1 -> v3 by simply saying there are no transitions'''	
transition_matrix[1,3] = 0

'''Make the LP problem'''
lp_problem = _pulp.LpProblem("Mincost Brain",_pulp.LpMinimize)

sequence = []
for i in range(0,dimension):
	sequence += str(i)

rows = sequence
columns = sequence
'''
First, lets specify the scaling factor variables.
Lets make it simple and have a scaling factor for each edge
'''
scaling_factors = _pulp.LpVariable.dicts("scaling_factors",(rows,columns),lowBound=1,cat="Continuous")

'''
Next up, lets create the flow variables.
Again, for simplicity, one for each edge (regardless of capacity)
'''
flow_values = _pulp.LpVariable.dicts("flow_values",(rows,columns),lowBound=0,cat="Continuous")


'''
Now its time to add the constraints
First, the objective function
in this case, just use lpsum over scaling_factors which takes in the list of variables

scaling_factors_list = a list of all of the scaling factor variables in this case
e.g.,
lp_problem += scaling_factor[0][0] + scaling_factors[0][1] + .... + scaling_fators[n][n]
'''

'''
Now lets add the Flow constraints

for each flow variable
lp_problem += flow_variable[i][j] <= scaling_factor[i][j] * transition_matrix[i][j]
'''


'''
Finally, lets solve it!

lp_problem.solve()
'''
