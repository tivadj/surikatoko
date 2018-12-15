from typing import Dict, Tuple, List, Optional

class Point2i:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
    def __str__(self):
        return "[{},{}]".format(self.x,self.y)

class Size2i:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
    def __str__(self):
        return "[{},{}]".format(self.width, self.height)

class Rect2i:
    def __init__(self, x, y, width, height):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)
    def Right(self):
        return self.x + self.width
    def Bottom(self):
        return self.y + self.height
    def TL(self):
        return Point2i(self.x,self.y)
    def BR(self):
        return Point2i(self.x+self.width,self.y+self.height)
    def __str__(self):
        return "[{},{},{},{}]".format(self.x,self.y, self.width,self.height)

def IntersectRects(a:Rect2i, b:Rect2i) -> Optional[Rect2i]:
    left = a
    right = b
    if left.x > right.x:
        left = b
        right = a

    if left.Right() <= right.x:
        return None

    top = a
    bot = b
    if top.y > bot.y:
        top = b
        bot = a

    if top.Bottom() <= bot.y:
        return None

    # now, there is certainly some non-empty crossing
    x1 = right.x
    x2 = min(a.Right(), b.Right())
    y1 = bot.y
    y2 = min(a.Bottom(), b.Bottom())
    return Rect2i(x1, y1, x2-x1, y2-y1)
