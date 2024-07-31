import numpy as np

def floyrd_warshall(gridSize):
    # Initialize the graph of the grid in terms of distances
    dist = np.zeros((gridSize * gridSize, gridSize * gridSize))
    # Initialize the last location to travel to
    l_node = np.zeros((gridSize * gridSize, gridSize * gridSize))

    for i in range(gridSize):
        for j in range(gridSize):
            for k in range(gridSize):
                for l in range(gridSize):
                    if (i == k and (j == l -1 or j == l + 1)) or (j == l and (i == k -1 or i == k + 1)):
                        dist[i * gridSize + j, k * gridSize + l] = 1
                    elif (i == k and j == l):
                        dist[i * gridSize + j, k * gridSize + l] = 0
                    else:
                        dist[i * gridSize + j, k * gridSize + l] = float("inf")

    for i in range(gridSize * gridSize):
        for j in range(gridSize * gridSize):
            if i == j:
                l_node[i, j] = 0
            elif dist[i, j] != float("inf"):
                l_node[i, j] = i
            else:
                l_node[i, j] = -1

    # Floyd-Warshall's algorithm
    for k in range(gridSize * gridSize):                                  #This is the starting location x, y
        for i in range(gridSize * gridSize):                              #This is the ending location x, y
            for j in range(gridSize * gridSize):                          #This is the mid location x, y
                if (dist[i, k] is not float("inf") and dist[k, j] is not float("inf")):
                    if (dist[i, j] > dist[i, k] + dist[k, j]):
                        dist[i, j] =  dist[i, k] + dist[k, j]
                        l_node[i, j] = l_node[k, j]


    return dist, l_node