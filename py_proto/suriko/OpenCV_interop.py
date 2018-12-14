from suriko.geom_elements import Point2i

# Converts to opencv 2D ponit in a form of a tuple.
def ocv(p):
    if isinstance(p, Point2i):
        return (p.x, p.y)
    return (p[1], p[0])
