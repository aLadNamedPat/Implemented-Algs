from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
#We can use shapely to deal with our geometry problems </3
class Obstacle:
    def __init__(self, *args):
        self.args = args
        self.polygon_coordinates = []
    def generate_polygon(self):
        for coords in self.args:
            self.polygon_coordinates.append(coords)
        return Polygon(self.polygon_coordinates)
    
