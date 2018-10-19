Run C++ demos:
```
demo-circle-grid.exe --logtostderr -v 4
demo-dino.exe --logtostderr -v 4 --testdata ../../../../testdata
```
Davison's MonoSlam implementation.

Implementation of MonoSlam by Andrew Davison https://www.doc.ic.ac.uk/~ajd/Scene/index.html
The algorithm uses Kalman filter to estimate camera's location and map features (salient points).
The algorithm is described in the book "Structure from Motion using the Extended Kalman Filter" Civera 2011.

The uncertainty ellipsoids of salient points are rendered with different color to distinguish:
Green - new salient points, added to the map;
Red - the salient point is matched to the one from previous frame.
Almost spherical ellipsoids are drawn with two extra strokes on the flanks.