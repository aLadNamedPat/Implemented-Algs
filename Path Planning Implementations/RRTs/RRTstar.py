from Grid import Grid
from SpecialPoint import SpecialPoint
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from Obstacles import Obstacle
# Initialize matplotlib plot
plt.ion()  # Turn on interactive plotting
fig, ax = plt.subplots()

# Assuming you've already run your RRT algorithm and have the final_trajectory
# as well as the obstacles list from your code snippet
obstacle1 = Obstacle((10, 1), (12, 1), (12,15), (10,15))
obstacles = [obstacle1]
new_obs = [[(10, 1), (12, 1), (12,15), (10,15)]]
grid = Grid(20, 20, obstacles)
nn_dist = 0.8
start_node = SpecialPoint(6, 8, 0)
end_node = SpecialPoint(18, 2, 0)

# Plot the start and end points
ax.plot(start_node.x, start_node.y, 'go', label='Start')  # Green dot for start
ax.plot(end_node.x, end_node.y, 'ro', label='End')  # Red dot for end

for obstacle in obstacles:
    polygon = obstacle.generate_polygon()  # Assuming this returns coordinates of the polygon
    ax.add_patch(plt.Polygon(polygon.exterior.coords, color='k'))  # Assuming polygon is a Shapely polygon

#How does RRT* work?
#If we want a solution that is optimal, then we would use RRTstar
#Choose a random node, find the nearest neighbor, then place a node at the nearest node on the
#line to the nearest node (Node selection is the same as RRT)
#Where the node is connected is important as RRT* will perform "reconnections" with nodes
#inside its search radius such that the distance of the tree is then optimized by the connection.

class RRTstar:
    def __init__(self, sample_distance, start_node, end_node, grid, nn_dist):
        self.sd = sample_distance
        self.start_node = start_node
        self.end_node = end_node
        self.grid = grid
        self.nodes_stored = [start_node]
        self.end_generation = False
        self.nn_dist = nn_dist

    def find_nearest_neighbor(self, point):
        shortest_dist = self.grid.x_lim * self.grid.y_lim
        node_to_use = None
        for node in self.nodes_stored:
            dist = self.find_dist(point, node)
            if (dist < shortest_dist):
                shortest_dist = dist
                node_to_use = node
        return node_to_use
    
    def find_lower_cost(self, point):
        for node in self.nodes_stored:
            dist = self.find_dist(point, node)
            current_cost = point.cost
            if (dist < self.nn_dist and node.cost + dist < current_cost and not self.intersection_with_obstacle(node, point)):
                point.lP = node
                point.cost = node.cost + dist
        return point

    def intersection_with_obstacle(self, point1, point2):
        return self.grid.intersection_with_obstacle(point1, point2)
    
    def inObstacle(self, point):
        return self.grid.in_obstacle(point)
    
    def find_dist(self, point1, point2):
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
    
    def outside_bound(self, point):
        if (point.x < 0 or point.y < 0 or point.x > self.grid.x_lim or point.y > self.grid.y_lim):
            return True
        return False
    
    def generate_next_point(self, bias, min_dist):
        if (random.random() < bias):
            next_point = self.end_node
        else:
            next_point = SpecialPoint(random.random() * self.grid.x_lim, random.random() * self.grid.y_lim)
        nearest_neighbor = self.find_nearest_neighbor(next_point)
        if (self.find_dist(next_point, nearest_neighbor) < self.sd):
            point_to_append = SpecialPoint(next_point.x, next_point.y, 
                                           nearest_neighbor.cost + self.find_dist(next_point, nearest_neighbor),
                                           nearest_neighbor)
        else:
            point_to_append = SpecialPoint(nearest_neighbor.x + self.sd / self.find_dist(next_point, nearest_neighbor)
                                    * (next_point.x - nearest_neighbor.x), 
                                    nearest_neighbor.y + self.sd / self.find_dist(next_point, nearest_neighbor)
                                    * (next_point.y - nearest_neighbor.y), 0, nearest_neighbor)
            point_to_append.cost = nearest_neighbor.cost + self.find_dist(point_to_append, nearest_neighbor)
        point_to_append = self.find_lower_cost(point_to_append)
        if not (self.inObstacle(point_to_append) or self.intersection_with_obstacle(point_to_append.lP, point_to_append) or self.outside_bound(next_point)):
            self.nodes_stored.append(point_to_append)
            if (self.find_dist(self.end_node, point_to_append) < min_dist and not self.intersection_with_obstacle(point_to_append, self.end_node)):
                self.end_generation = True
            return point_to_append
        return None
    
    def generate_graphs(self, bias, min_dist):
        while (not self.end_generation):
            last_point = self.generate_next_point(bias, min_dist)
            if last_point:
                # Plot the new point
                ax.plot(last_point.x, last_point.y, 'bo')  # Blue dot for new points

                # Draw a line from the new point to its nearest neighbor
                nearest_neighbor = last_point.lP
                ax.plot([last_point.x, nearest_neighbor.x], [last_point.y, nearest_neighbor.y], 'b-')

                # Redraw the plot with the new point and line
                plt.draw()
                plt.pause(0.01)  # Pause briefly to show updates
        sp = last_point
        final_trajectory = [(self.end_node.x, self.end_node.y)]
        # Turn off interactive plotting
        plt.ioff()

        while (sp.lP != None):
            final_trajectory.append((sp.x, sp.y))
            sp = sp.lP
        final_trajectory.append((self.start_node.x , self.start_node.y))
        final_trajectory = final_trajectory[::-1]
        return last_point, final_trajectory
    
rrt = RRTstar(1.0, start_node, end_node, grid, nn_dist)
final_trajectory = rrt.generate_graphs(0.9, 0.6)[1]

