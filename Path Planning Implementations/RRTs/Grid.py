from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon

class Grid:
    def __init__(self, x_lim, y_lim, obstacles):
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.obstacles = [obs.generate_polygon() for obs in obstacles]
        # Create a MultiPolygon object for all obstacles
        self.obstacles_multipolygon = MultiPolygon(self.obstacles)

    def intersection_with_obstacle(self, point1, point2):
        linewp = LineString([(point1.x, point1.y), (point2.x, point2.y)])
        # Check for intersection with all obstacles at once
        return linewp.intersects(self.obstacles_multipolygon)

    def in_obstacle(self, point):
        p1 = Point(point.x, point.y)
        # Check if the point is inside any of the obstacles
        return self.obstacles_multipolygon.contains(p1)
