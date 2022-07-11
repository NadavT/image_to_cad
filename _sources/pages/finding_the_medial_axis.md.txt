# Finding the medial axis

## Goal

We would like to find the medial axis of a given image. To do so, we use a couple techniques in order to clean up the image and find the medial axis.

Currently our approach uses the following techniques:

1. **Image Preprocessing**: We use a series of filters to clean up the image before running further algorithms.
2. **Edge detection**: We use opencv to detect edges in the image (findContours), we also apply canny edge detection to find the edges of the image
3. **Voronoi Diagram**: We calculate and use a voronoi diagram (as a graph).
4. **Final filtering**: We apply a series of algorithms to clean up the voronoi diagram and find the medial axis.

for example:

![Example](assets/medial_axis.png "Example")

The medial axis of the image is in red/pink.
