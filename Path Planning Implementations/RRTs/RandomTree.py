from Grid import Grid
from Point import Point
import math
import random
from Obstacles import Obstacle
import matplotlib.pyplot as plt

plt.ion()  # Turn on interactive plotting
fig, ax = plt.subplots()

#Sampling based algorithm
#We want to generate from the starting position to the ending position
#While we generate goals as we enter there
obstacle1 = Obstacle((10, 1), (12, 1), (12,15), (10,15))
obstacles = [obstacle1]
new_obs = [[(10, 1), (12, 1), (12,15), (10,15)]]
grid = Grid(20, 20, obstacles)
starting_point = Point(6, 8)
ending_point = Point(18, 2)

for obstacle in obstacles:
    polygon = obstacle.generate_polygon()  # Assuming this returns coordinates of the polygon
    ax.add_patch(plt.Polygon(polygon.exterior.coords, color='k'))  # Assuming polygon is a Shapely polygon

#How do Random Trees work? 
#Start with a starting node, place a random node anywhere in the state space
#Connect the random node to the nearest node in the state space (which is just the start node)
#Specify a maximum node that the new node can be to the new node.
#Draw the actual node on the nearest point that is on the line to the new node generated
#Randomly selected node on the opposite side of the goal won't affect the algorithm
#Because the node would be connected to the original node
#As the number of points generated reaches some threshold close to the goal, then we have
#Reached a good solution, but not optimal

class RandomTree:
    def __init__(self, sample_distance, start_node, end_node, grid):
        self.sd = sample_distance
        self.start_node = start_node
        self.end_node = end_node
        self.grid = grid
        self.nodes_stored = [start_node]
        self.end_generation = False

    def find_nearest_neighbor(self, point):
        shortest_dist = self.grid.x_lim * self.grid.y_lim
        node_to_use = None
        for node in self.nodes_stored:
            dist = self.find_dist(point, node)
            if (dist < shortest_dist):
                shortest_dist = dist
                node_to_use = node
        return node_to_use

    def intersection_with_obstacle(self, point1, point2):
        return self.grid.intersection_with_obstacle(point1, point2)
    
    def inObstacle(self, point):
        return self.grid.in_obstacle(point)
    
    def find_dist(self, point1, point2):
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
    
    def outside_bound(self, point):
        if (point.x < 0 or point.y < 0 or point.x > self.grid.x_lim or point.y > self.grid.y_lim):
            return True
        return False
    
    def generate_next_point(self, threshold):
        next_point = Point(random.random() * self.grid.x_lim, random.random() * self.grid.y_lim)
        nearest_neighbor = self.find_nearest_neighbor(next_point)
        if (self.find_dist(next_point, nearest_neighbor) < self.sd):
            point_to_append = Point(next_point.x, next_point.y, nearest_neighbor)
        else:
            point_to_append = Point(nearest_neighbor.x + self.sd / self.find_dist(next_point, nearest_neighbor)
                                    * (next_point.x - nearest_neighbor.x), 
                                    nearest_neighbor.y + self.sd / self.find_dist(next_point, nearest_neighbor)
                                    * (next_point.y - nearest_neighbor.y),
                                    nearest_neighbor)
        
        if not (self.inObstacle(point_to_append) or self.intersection_with_obstacle(nearest_neighbor, point_to_append) or self.outside_bound(next_point)):
            self.nodes_stored.append(point_to_append)
            if (self.find_dist(self.end_node, point_to_append) < 0.5 and not self.intersection_with_obstacle(point_to_append, self.end_node)):
                self.end_generation = True
            return point_to_append
        return None
    
    def generate_graphs(self, threshold):
        plt.scatter(self.start_node.x, self.start_node.y, c='green', label='Start')
        plt.scatter(self.end_node.x, self.end_node.y, c='red', label='End')
        while (not self.end_generation):
            last_point = self.generate_next_point(threshold)
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
        while (sp.lP != None):
            final_trajectory.append((sp.x, sp.y))
            sp = sp.lP
        final_trajectory.append((self.start_node.x , self.start_node.y))
        final_trajectory = final_trajectory[::-1]
        return last_point, final_trajectory
    
rt = RandomTree(1, starting_point, ending_point, grid)

final_trajectory = rt.generate_graphs(0.9)[1]
