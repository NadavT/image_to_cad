import math
import sys
import pyvoronoi
import cv2
import numpy as np
from tqdm import tqdm
import networkx as nx
image_path = "resources/xPhys.ppm"
# image_path = "resources/square.png"


def load_image(path):
    # Get image
    image = cv2.imread(path)
    image = cv2.resize(
        image, (image.shape[1] * 4, image.shape[0] * 4), interpolation=cv2.INTER_NEAREST_EXACT)
    to_concatenate = np.zeros((image.shape[0], 1, 3), dtype=np.uint8) * 255
    # image = np.concatenate((to_concatenate, image), axis=1)
    # image = np.concatenate((image, to_concatenate), axis=1)
    # to_concatenate = np.zeros((1, image.shape[1], 3), dtype=np.uint8) * 255
    # image = np.concatenate((to_concatenate, image), axis=0)
    # image = np.concatenate((image, to_concatenate), axis=0)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edges
    edged = cv2.Canny(gray, 30, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bounding = (
        np.array([
            np.array([np.array([1, 1])]),
            np.array([np.array([1, image.shape[0]-1])]),
            np.array([np.array([image.shape[1]-1, image.shape[0]-1])]),
            np.array([np.array([image.shape[1]-1, 1])]),
            # np.array([np.array([image.shape[1]-1, 1])]),
        ]),
        np.array([
            np.array([np.array([1, 1])]),
            np.array([np.array([image.shape[1]-1, 1])]),
        ]),
    )
    bounding = (
        np.array([np.array([np.array([0, i])])
                 for i in range(image.shape[0] - 1)]),
        np.array([np.array([np.array([i, 0])])
                 for i in range(image.shape[1] - 1)]),
        np.array([np.array([np.array([image.shape[1] - 1, i])])
                 for i in range(image.shape[0] - 1)]),
        np.array([np.array([np.array([i, image.shape[0] - 1])])
                 for i in range(image.shape[1] - 1)]),
    )
    contours = contours + bounding
    return image, gray, contours, hierarchy


def calculate_voronoi(contours, width, height):
    # Create voronoi diagram
    image_cont = np.ones((height, width)) * 255
    print("\t Creating voronoi diagram")
    cv2.namedWindow('cont', cv2.WINDOW_NORMAL)
    pv = pyvoronoi.Pyvoronoi(1)
    segments = []
    for contour in contours:
        prev = None
        for line in contour:
            for vertex in line:
                if prev is not None:
                    pv.AddSegment([[prev[0], prev[1]], [vertex[0], vertex[1]]])
                    cv2.line(image_cont, (prev[0], prev[1]), (vertex[0], vertex[1]),
                             (0, 0, 0), thickness=1)
                    segments.append(
                        ([prev[0], prev[1]], [vertex[0], vertex[1]]))
                prev = vertex
        # cv2.waitKey(0)

    print("\t Finished adding segments")
    pv.Construct()
    print("\t Finished construction")
    return pv, segments


def check_mask(image, x, y):
    if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
        return False
    return image[y][x][2] == 255


def distance_to_edge(point, edge_start, edge_end):
    return abs((edge_end[0] - edge_start[0])*(edge_start[1] - point.Y) - (edge_start[0] - point.X)*(edge_end[1] -
                                                                                                    edge_start[1])) / math.sqrt((edge_end[0] - edge_start[0])**2 + (edge_end[1] - edge_start[1])**2)


def is_local_maxima_edge(cell, vertices, edges, segments, edge_index):
    edge_to_check = edges[edge_index]
    incident_segments = segments[cell.site]
    start_distance = distance_to_edge(
        vertices[edge_to_check.start], incident_segments[0], incident_segments[1])
    end_distance = distance_to_edge(
        vertices[edge_to_check.end], incident_segments[0], incident_segments[1])
    if max(start_distance, end_distance) <= 1 or abs(end_distance - start_distance) / max(start_distance, end_distance) > 0.5:
        return False
    return True


def is_local_maxima_point(cell, vertices, edges, segments, edge_index):
    edge_to_check = edges[edge_index]
    start = (vertices[edge_to_check.start].X,
             vertices[edge_to_check.start].Y)
    site = (vertices[cell.site].X, vertices[cell.site].Y)
    edge_to_check_distance = math.dist(start, site)
    for edge in cell.edges:
        start = (vertices[edges[edge].start].X,
                 vertices[edges[edge].start].Y)
        distance = math.dist(start, site)
        if edge_to_check_distance < distance:
            print("not local maxima")
            return False
    return True


def is_local_maxima(cell, vertices, edges, segments, edge_index):
    return True
    # return is_local_maxima_edge(cell, vertices, edges, segments, edge_index)


def draw_voronoi(image, pv, segments):
    # Create a blank image in numpy of width and height of image
    width, height = image.shape[1], image.shape[0]
    image_vor = np.ones((height, width, 3)) * 255
    image_cells = np.ones((height, width, 3)) * 255
    edges = pv.GetEdges()
    vertices = pv.GetVertices()
    cells = pv.GetCells()
    colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255),
              (255, 0, 255), (255, 255, 0), (255, 0, 0), (0, 0, 0), (0, 128, 255), (128, 255, 0)]
    cv2.namedWindow('voronoi', cv2.WINDOW_NORMAL)
    cv2.namedWindow('comb', cv2.WINDOW_NORMAL)
    cv2.namedWindow('cells', cv2.WINDOW_NORMAL)
    graph = nx.Graph()
    line_thickness = 1
    color = 0
    edge_color = 0
    for cell in tqdm(cells):
        color = (color + 1) % len(colors)
        if not cell.is_open:
            # print(pv.RetrieveSegment(cell), cell.is_degenerate)
            # cv2.imshow("voronoi", image_vor)
            # cv2.imshow("comb", image)
            # cv2.resizeWindow('voronoi', 600, 600)
            # cv2.resizeWindow('comb', 600, 600)
            # cv2.waitKey(0)
            cell_vertices = []
            incident_edge = edges[cell.site]
            for i in range(len(cell.edges)):
                edge_color = (edge_color + 1) % len(colors)
                edge = edges[cell.edges[i]]
                startVertex = vertices[edge.start]
                endVertex = vertices[edge.end]
                # if not edge.is_primary:
                #     continue

                if startVertex != -1 and endVertex != -1:
                    distance = math.dist([startVertex.X, startVertex.Y], [
                        endVertex.X, endVertex.Y])
                    if(edge.is_linear == True):
                        if (not math.isinf(vertices[edge.start].X) and not math.isinf(vertices[edge.end].X)
                                and not math.isinf(vertices[edge.start].Y) and not math.isinf(vertices[edge.end].Y)
                                and not math.isnan(vertices[edge.start].X) and not math.isnan(vertices[edge.end].X)
                                and not math.isnan(vertices[edge.start].Y) and not math.isnan(vertices[edge.end].Y)):
                            start_vertex = (int(vertices[edge.start].X), int(
                                vertices[edge.start].Y))
                            end_vertex = (int(vertices[edge.end].X),
                                          int(vertices[edge.end].Y))
                            if check_mask(image, start_vertex[0], start_vertex[1]) and check_mask(image, end_vertex[0], end_vertex[1]):
                                if is_local_maxima(cell, vertices, edges, segments, cell.edges[i]):
                                    cv2.line(image_vor, start_vertex, end_vertex,
                                             colors[edge_color], thickness=line_thickness)
                                    cv2.line(image, start_vertex, end_vertex,
                                             (0, 0, 255), thickness=line_thickness)
                                    if start_vertex != end_vertex:
                                        graph.add_node(
                                            start_vertex, x=start_vertex[0], y=start_vertex[1])
                                        graph.add_node(
                                            end_vertex, x=end_vertex[0], y=end_vertex[1])
                                        graph.add_edge(
                                            start_vertex, end_vertex, length=math.dist(start_vertex, end_vertex))
                                    cell_vertices.append(
                                        start_vertex)
                    elif distance > 0:
                        try:
                            points = pv.DiscretizeCurvedEdge(
                                cell.edges[i], distance, 1)
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
                                        if check_mask(image, start_vertex[0], start_vertex[1]) and check_mask(image, end_vertex[0], end_vertex[1]):
                                            if is_local_maxima(cell, vertices, edges, segments, cell.edges[i]):
                                                cv2.line(image_vor, start_vertex, end_vertex,
                                                         colors[edge_color], thickness=line_thickness)
                                                cv2.line(image, start_vertex, end_vertex,
                                                         (0, 0, 255), thickness=line_thickness)
                                                if start_vertex != end_vertex:
                                                    graph.add_node(
                                                        start_vertex, x=start_vertex[0], y=start_vertex[1])
                                                    graph.add_node(
                                                        end_vertex, x=end_vertex[0], y=end_vertex[1])
                                                    graph.add_edge(
                                                        start_vertex, end_vertex, length=math.dist(start_vertex, end_vertex))
                                                cell_vertices.append(
                                                    start_vertex)
                                prev = p
                        except pyvoronoi.UnsolvableParabolaEquation:
                            print("UnsolvableParabolaEquation")
                        except ZeroDivisionError:
                            print("ZeroDivisionError")
            if len(cell_vertices) > 0:
                cv2.fillPoly(image_cells, pts=[np.array(
                             cell_vertices)], color=colors[color])
                # start_vertex = [int(vertices[incident_edge.start].X), int(
                #     vertices[incident_edge.start].Y)]
                # end_vertex = [int(vertices[incident_edge.end].X),
                #               int(vertices[incident_edge.end].Y)]
                # cv2.line(image_cells, start_vertex, end_vertex,
                #          (0, 0, 0), thickness=line_thickness)

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

    for x, y in graph.nodes():
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if (i != x or j != y) and (i, j) in graph.nodes():
                    edges = nx.bfs_edges(graph, (x, y), depth_limit=3)
                    nodes = [v for u, v in edges]
                    if (i, j) not in nodes:
                        graph.add_edge(
                            (x, y), (i, j), length=math.dist((x, y), (i, j)))

    cv2.imshow("voronoi", image_vor)
    cv2.imshow("comb", image)
    cv2.imshow("cells", image_cells)
    cv2.imwrite("results/voronoi.png", image_vor)
    cv2.imwrite("results/combined.png", image)
    # cv2.resizeWindow('voronoi', 600, 600)
    # cv2.resizeWindow('comb', 600, 600)
    print("Done")
    return graph


def check_for_flower(image, i, j):
    if i > 0 and j > 0 and i < image.shape[0] - 1 and j < image.shape[1] - 1:
        left = image[i-1][j]
        right = image[i+1][j]
        up = image[i][j-1]
        down = image[i][j+1]
        if left[0] == 0 and left[1] == 0 and left[2] == 255 and \
                right[0] == 0 and right[1] == 0 and right[2] == 255 and \
                up[0] == 0 and up[1] == 0 and up[2] == 255 and \
                down[0] == 0 and down[1] == 0 and down[2] == 255:
            return True
        return False


def check_surrounding(image, i, j, new_red_pixels):
    amount = 0
    check_up_left = True
    check_up_right = True
    check_down_left = True
    check_down_right = True
    up = False
    down = False
    left = False
    right = False
    up_left = False
    up_right = False
    down_left = False
    down_right = False
    if i > 0:
        pixel = image[i-1][j]
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
            amount += 1
            check_up_left = False
            check_up_right = False
            up = True
    if i < image.shape[0]-1:
        pixel = image[i+1][j]
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
            amount += 1
            check_down_left = False
            check_down_right = False
            down = True
    if j > 0:
        pixel = image[i][j-1]
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
            amount += 1
            check_up_left = False
            check_down_left = False
            left = True
    if j < image.shape[1]-1:
        pixel = image[i][j+1]
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
            amount += 1
            check_up_right = False
            check_down_right = False
            right = True
    if i > 0 and j > 0:
        pixel = image[i-1][j-1]
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
            if up and left:
                amount -= 1
            elif not up and not left:
                amount += 1
            up_left = True
    if i > 0 and j < image.shape[1]-1:
        pixel = image[i-1][j+1]
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
            if up and right:
                amount -= 1
            elif not up and not right:
                amount += 1
            up_right = True
    if i < image.shape[0]-1 and j > 0:
        pixel = image[i+1][j-1]
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
            if down and left:
                amount -= 1
            elif not down and not left:
                amount += 1
            down_left = True
    if i < image.shape[0]-1 and j < image.shape[1]-1:
        pixel = image[i+1][j+1]
        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255:
            if down and right:
                amount -= 1
            elif not down and not right:
                amount += 1
            down_right = True
    # for col in range(max(i - 1, 0), min(i + 2, image.shape[0])):
    #     for row in range(max(j - 1, 0), min(j + 2, image.shape[1])):
    #         if image[col][row][0] == 0 and image[col][row][1] == 0 and image[col][row][2] == 255:  # Red
    #             amount += 1
    if amount < 2:
        image[i, j] = [255, 255, 255]
        return True
    else:
        if up_left and up_right and check_for_flower(image, i - 1, j):
            image[i-1][j] = [0, 0, 255]
            new_red_pixels.append((i-1, j))
        if down_left and down_right and check_for_flower(image, i + 1, j):
            image[i+1][j] = [0, 0, 255]
            new_red_pixels.append((i+1, j))
        if up_left and down_left and check_for_flower(image, i, j - 1):
            image[i][j-1] = [0, 0, 255]
            new_red_pixels.append((i, j-1))
        if up_right and down_right and check_for_flower(image, i, j + 1):
            image[i][j+1] = [0, 0, 255]
            new_red_pixels.append((i, j+1))
    return False


def remove_hanging(image):
    changed = True
    cv2.namedWindow('processed', cv2.WINDOW_NORMAL)
    red_pixels = []
    print("Get red pixels")
    for i in tqdm(range(image.shape[0])):
        for j in range(image.shape[1]):
            if image[i][j][0] == 0 and image[i][j][1] == 0 and image[i][j][2] == 255:  # Red
                if i > 1 and j > 1 and i < image.shape[0]-2 and j < image.shape[1]-2:
                    red_pixels.append((i, j))

    print("Removing layers")
    epoch = 0
    while changed and len(red_pixels) > 0:
        print(f"Epoch {epoch}")
        new_red_pixels = []
        if epoch % 10 == 0:
            cv2.imshow('processed', image)
            cv2.waitKey(0)
        changed = False
        for i, j in tqdm(reversed(red_pixels), total=len(red_pixels)):
            if check_surrounding(image, i, j, new_red_pixels):
                changed = True
            else:
                new_red_pixels.append((i, j))
        red_pixels = new_red_pixels
        epoch += 1
    cv2.imshow('processed', image)
    cv2.waitKey(0)


def remove_hanging_by_graph(original_image, graph):
    cv2.namedWindow('processed', cv2.WINDOW_NORMAL)
    changed = True
    epoch = 0
    while changed:
        new_graph = graph.copy()
        changed = False
        leafs = [node for node in graph.nodes if graph.degree(node) == 1]
        print(f"Epoch {epoch}")
        # if epoch % 10 == 0:
        #     image = original_image.copy()
        #     for start, end in tqdm(graph.edges()):
        #         cv2.line(image, start, end, (0, 0, 255), thickness=1)
        #     cv2.imshow('processed', image)
        #     cv2.waitKey(0)
        for leaf in tqdm(leafs):
            to_remove = [leaf]
            parent, data = list(graph[leaf].items())[0]
            prev = leaf
            leaf_length = data['length']
            while graph.degree(parent) == 2:
                to_remove.append(parent)
                for node, data in graph[parent].items():
                    if node != prev:
                        leaf_length += data['length']
                        prev = parent
                        parent = node
                        break
            if leaf_length < 100:
                changed = True
                for node in to_remove:
                    new_graph.remove_node(node)
        graph = new_graph
        epoch += 1
    image = original_image.copy()
    for start, end in tqdm(graph.edges()):
        cv2.line(image, start, end, (0, 0, 255), thickness=1)
    cv2.imshow('processed', image)
    cv2.imwrite('results/final_result.png', image)
    print("Finished!")


def main():
    global image_path
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
    print("Loading image...")
    image, grayscale, contours, _ = load_image(image_path)
    original = image.copy()
    cv2.imshow("Original", original)
    cv2.imwrite("results/original.png", original)
    cv2.drawContours(image, contours, -1, (0xff, 0, 0), 1)
    cv2.imshow('contours', image)
    cv2.imwrite("results/contours.png", image)
    print("Press any key to continue to calculation...")
    cv2.waitKey(0)
    pv, segments = calculate_voronoi(contours, image.shape[1], image.shape[0])
    print("Finished calculating voronoi")
    print("drawing...")
    graph = draw_voronoi(image, pv, segments)
    # remove_hanging(image)
    remove_hanging_by_graph(original, graph)
    while cv2.waitKey(0) != 27:  # Escape key
        pass
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
