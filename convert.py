import math
import sys
import pyvoronoi
import cv2
import numpy as np
from tqdm import tqdm
image_path = "resources/xPhys.ppm"
# image_path = "resources/square.png"


def load_image(path):
    # Get image
    image = cv2.imread(path)
    image = cv2.resize(
        image, (image.shape[1] * 4, image.shape[0] * 4), interpolation=cv2.INTER_NEAREST_EXACT)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edges
    edged = cv2.Canny(gray, 30, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return image, gray, contours, hierarchy


def calculate_voronoi(contours, width, height):
    # Create voronoi diagram
    image_cont = np.ones((height, width)) * 255
    print("\t Creating voronoi diagram")
    cv2.namedWindow('cont', cv2.WINDOW_NORMAL)
    pv = pyvoronoi.Pyvoronoi(1)
    for contour in contours:
        prev = None
        first = None
        for line in contour:
            for vertex in line:
                if first is None:
                    first = vertex
                if prev is not None:
                    pv.AddSegment([[prev[0], prev[1]], [vertex[0], vertex[1]]])
                    cv2.line(image_cont, (prev[0], prev[1]), (vertex[0], vertex[1]),
                             (0, 0, 0), thickness=1)
                prev = vertex
        # if first is not None:
        #     pv.AddSegment(
        #         [[contour[-1][-1][0], contour[-1][-1][1]], [first[0], first[1]]])
        #     cv2.line(image_cont, (contour[-1][-1][0], contour[-1][-1][1]), (first[0], first[1]),
        #              (0, 0, 0), thickness=1)
        cv2.imshow("cont", image_cont)
        # cv2.waitKey(0)

    print("\t Finished adding segments")
    pv.Construct()
    print("\t Finished construction")
    return pv


def distance(a, b):
    return math.dist(a, b)


def check_mask(image, x, y):
    if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
        return False
    return image[y][x] != 0


def draw_voronoi(image, pv, grayscale):
    # Create a blank image in numpy of width and height of image
    width, height = image.shape[1], image.shape[0]
    image_vor = np.ones((height, width)) * 255
    edges = pv.GetEdges()
    vertices = pv.GetVertices()
    cells = pv.GetCells()

    cv2.namedWindow('voronoi', cv2.WINDOW_NORMAL)
    cv2.namedWindow('comb', cv2.WINDOW_NORMAL)
    line_thickness = 1
    for cell in tqdm(cells):
        if not cell.is_open:
            # print(pv.RetrieveSegment(cell), cell.is_degenerate)
            # cv2.imshow("voronoi", image_vor)
            # cv2.imshow("comb", image)
            # cv2.resizeWindow('voronoi', 600, 600)
            # cv2.resizeWindow('comb', 600, 600)
            # cv2.waitKey(0)
            for i in range(len(cell.edges)):
                edge = edges[cell.edges[i]]
                startVertex = vertices[edge.start]
                endVertex = vertices[edge.end]
                cell_vertices = []
                # if not edge.is_primary:
                #     continue

                if startVertex != -1 and endVertex != -1:
                    max_distance = distance([startVertex.X, startVertex.Y], [
                                            endVertex.X, endVertex.Y]) / 10
                    if(edge.is_linear == True):
                        if (not math.isinf(vertices[edge.start].X) and not math.isinf(vertices[edge.end].X)
                                and not math.isinf(vertices[edge.start].Y) and not math.isinf(vertices[edge.end].Y)
                                and not math.isnan(vertices[edge.start].X) and not math.isnan(vertices[edge.end].X)
                                and not math.isnan(vertices[edge.start].Y) and not math.isnan(vertices[edge.end].Y)):
                            start_vertex = (int(vertices[edge.start].X), int(
                                vertices[edge.start].Y))
                            end_vertex = (int(vertices[edge.end].X),
                                          int(vertices[edge.end].Y))
                            if check_mask(grayscale, start_vertex[0], start_vertex[1]) and check_mask(grayscale, end_vertex[0], end_vertex[1]):
                                cv2.line(image_vor, start_vertex, end_vertex,
                                         (0, 255, 0), thickness=line_thickness)
                                cv2.line(image, start_vertex, end_vertex,
                                         (0, 0, 255), thickness=line_thickness)
                                cell_vertices.append(start_vertex)
                                cell_vertices.append(end_vertex)
                    elif max_distance > 0:
                        try:
                            points = pv.DiscretizeCurvedEdge(
                                cell.edges[i], max_distance)
                            prev = None
                            for p in points:
                                if prev is not None:
                                    if (not math.isinf(prev[0]) and not math.isinf(p[0])
                                            and not math.isinf(prev[1]) and not math.isinf(p[1])
                                            and not math.isnan(prev[0]) and not math.isnan(p[0])
                                            and not math.isnan(prev[1]) and not math.isnan(p[1])):
                                        start_vertex = (
                                            int(prev[0]), int(prev[1]))
                                        end_vertex = (int(p[0]), int(p[1]))
                                        if check_mask(grayscale, start_vertex[0], start_vertex[1]) and check_mask(grayscale, end_vertex[0], end_vertex[1]):
                                            cv2.line(image_vor, start_vertex, end_vertex,
                                                     (0, 255, 0), thickness=line_thickness)
                                            cv2.line(image, start_vertex, end_vertex,
                                                     (0, 0, 255), thickness=line_thickness)
                                            cell_vertices.append(start_vertex)
                                            cell_vertices.append(end_vertex)
                                prev = p
                        except pyvoronoi.UnsolvableParabolaEquation:
                            print("UnsolvableParabolaEquation")
                        except ZeroDivisionError:
                            print("ZeroDivisionError")

    # for edge in edges:
    #     if edge.start != -1 and edge.end != -1:
    #         start_vertex = (int(vertices[edge.start].X), int(
    #             vertices[edge.start].Y))
    #         end_vertex = (int(vertices[edge.end].X), int(vertices[edge.end].Y))
    #         max_distance = distance((start_vertex[0], start_vertex[1]), (
    #                                 end_vertex[0], end_vertex[1])) / 10
    #         if edge.is_linear:
    #             cv2.line(image_vor, start_vertex, end_vertex,
    #                      (0, 255, 0), thickness=line_thickness)
    #         else:
    #             print(max_distance)
    #             points = pv.DiscretizeCurvedEdge(edge, max_distance)

    cv2.imshow("voronoi", image_vor)
    cv2.imshow("comb", image)
    # cv2.resizeWindow('voronoi', 600, 600)
    # cv2.resizeWindow('comb', 600, 600)
    print("Done")
    while cv2.waitKey(0) != 27:  # Escape key
        pass


def main():
    global image_path
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
    print("Loading image...")
    image, grayscale, contours, _ = load_image(image_path)
    cv2.imshow("Original", image)
    cv2.drawContours(image, contours, -1, (0xff, 0, 0), 1)
    cv2.imshow('contours', image)
    print("Press any key to continue to calculation...")
    cv2.waitKey(0)
    pv = calculate_voronoi(contours, image.shape[1], image.shape[0])
    print("Finished calculating voronoi")
    print("drawing...")
    draw_voronoi(image, pv, grayscale)
    cv2.destroyAllWindows()


def get_contours():
    contours = []
    contours.append([[[10, 10], [10, 90]]])
    contours.append([[[10, 90], [90, 90]]])
    contours.append([[[90, 90], [90, 10]]])
    contours.append([[[90, 10], [10, 10]]])

    contours.append([[[20, 20], [20, 80]]])
    contours.append([[[20, 80], [80, 80]]])
    contours.append([[[80, 80], [80, 20]]])
    contours.append([[[80, 20], [20, 20]]])
    return contours


def test():
    width, height = 100, 100
    # contours = [[[[10, 10], [10, 100]], [[10, 100], [100, 100]],
    #              [[100, 100], [100, 10]], [[100, 10], [10, 10]]],
    #             [[[20, 20], [20, 90]], [[20, 90], [90, 90]],
    #              [[90, 90], [90, 20]], [[90, 20], [20, 20]]]]
    image = np.ones((height, width, 3), np.uint8) * 255
    contours = get_contours()
    for contour in contours:
        for line in contour:
            cv2.line(image, (line[0][0], line[0][1]), (line[1][0], line[1][1]),
                     (0, 0, 0), thickness=1)
    pv = calculate_voronoi(contours, width, height)
    print("Finished calculating voronoi")
    print("drawing...")
    draw_voronoi(image, pv)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    # test()
