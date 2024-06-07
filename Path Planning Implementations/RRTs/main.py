from RandomTree import RandomTree
from Grid import Grid
from Obstacles import Obstacle
from Point import Point
#Run the RRT program and visualize using matplotlib

obstacle1 = Obstacle((10, 1), (12, 1), (12,15), (10,15))
obstacles = [obstacle1]
grid = Grid(20, 20, obstacles)
starting_point = Point(6, 8)
ending_point = Point(18, 2)


rt = RandomTree(0.5, starting_point, ending_point, grid)
print(rt.generate_graphs(0.5)[1])