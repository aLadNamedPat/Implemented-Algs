class SpecialPoint:
    def __init__(self, x, y, cost = 0, lastPoint = None):
        self.x = x
        self.y = y
        self.lP = lastPoint
        self.cost = cost