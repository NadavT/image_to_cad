import math
import sys
import pyvoronoi
import cv2
import numpy as np
from tqdm import tqdm
import networkx as nx
import argparse

image_path = "resources/xPhys.ppm"


def load_image(path, scale_factor):
    # Get image
    image = cv2.imread(path)
    image = cv2.resize(
        image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST_EXACT)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edges
    edged = cv2.Canny(gray, 30, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
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
    print("\tCreating voronoi diagram")
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

    print("\tFinished adding segments")
    pv.Construct()
    print("\tFinished construction")
    return pv, segments


def check_mask(image, x, y):
    if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
        return False
    return image[y][x][2] == 255


def distance_to_edge(point, edge_start, edge_end):
    return abs((edge_end[0] - edge_start[0])*(edge_start[1] - point[1]) - (edge_start[0] - point[0])*(edge_end[1] -
                                                                                                      edge_start[1])) / math.sqrt((edge_end[0] - edge_start[0])**2 + (edge_end[1] - edge_start[1])**2)


def draw_voronoi(image, pv, segments, debug=False):
    if debug:
        # Create a blank image in numpy of width and height of image
        width, height = image.shape[1], image.shape[0]
        image_vor = np.ones((height, width, 3)) * 255
        image_cells = np.ones((height, width, 3)) * 255
        colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255),
                  (255, 0, 255), (255, 255, 0), (255, 0, 0), (0, 0, 0), (0, 128, 255), (128, 255, 0)]
        line_thickness = 1
        color = 0
    edges = pv.GetEdges()
    vertices = pv.GetVertices()
    cells = pv.GetCells()
    graph = nx.Graph()
    edge_color = 0
    for cell in tqdm(cells):
        if debug:
            color = (color + 1) % len(colors)
        if not cell.is_open:
            cell_vertices = []
            incident_segments = segments[cell.site]
            for i in range(len(cell.edges)):
                if debug:
                    edge_color = (edge_color + 1) % len(colors)
                edge = edges[cell.edges[i]]
                startVertex = vertices[edge.start]
                endVertex = vertices[edge.end]

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
                                if debug:
                                    cv2.line(image_vor, start_vertex, end_vertex,
                                             colors[edge_color], thickness=line_thickness)
                                    cv2.line(image, start_vertex, end_vertex,
                                             (0, 0, 255), thickness=line_thickness)
                                if start_vertex != end_vertex:
                                    graph.add_node(
                                        start_vertex, distance_to_source=distance_to_edge(start_vertex, incident_segments[0], incident_segments[1]))
                                    graph.add_node(
                                        end_vertex, distance_to_source=distance_to_edge(end_vertex, incident_segments[0], incident_segments[1]))
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
                                            if debug:
                                                cv2.line(image_vor, start_vertex, end_vertex,
                                                         colors[edge_color], thickness=line_thickness)
                                                cv2.line(image, start_vertex, end_vertex,
                                                         (0, 0, 255), thickness=line_thickness)
                                            if start_vertex != end_vertex:
                                                graph.add_node(
                                                    start_vertex, distance_to_source=distance_to_edge(start_vertex, incident_segments[0], incident_segments[1]))
                                                graph.add_node(
                                                    end_vertex, distance_to_source=distance_to_edge(end_vertex, incident_segments[0], incident_segments[1]))
                                                graph.add_edge(
                                                    start_vertex, end_vertex, length=math.dist(start_vertex, end_vertex))
                                            cell_vertices.append(
                                                start_vertex)
                                prev = p
                        except pyvoronoi.UnsolvableParabolaEquation:
                            print("UnsolvableParabolaEquation")
                        except ZeroDivisionError:
                            print("ZeroDivisionError")
            if debug and len(cell_vertices) > 0:
                cv2.fillPoly(image_cells, pts=[np.array(
                             cell_vertices)], color=colors[color])

    if debug:
        cv2.namedWindow('voronoi', cv2.WINDOW_NORMAL)
        cv2.namedWindow('comb', cv2.WINDOW_NORMAL)
        cv2.namedWindow('cells', cv2.WINDOW_NORMAL)
        cv2.imshow("voronoi", image_vor)
        cv2.imshow("comb", image)
        cv2.imshow("cells", image_cells)
        cv2.imwrite("results/voronoi.png", image_vor)
        cv2.imwrite("results/combined.png", image)
    return graph


def reduce_graph(graph, reduction_proximity):
    small_edges = dict()
    for u, v, data in graph.edges(data=True):
        if data['length'] < reduction_proximity:
            if graph.nodes[u]['distance_to_source'] >= graph.nodes[v]['distance_to_source']:
                small_edges[(u, v)] = True
            else:
                small_edges[(v, u)] = True
    total_len = len(small_edges)
    with tqdm(total=total_len) as pbar:
        while len(small_edges) > 0:
            if total_len - len(small_edges) > 0:
                pbar.update(total_len - len(small_edges))
            total_len = len(small_edges)
            for u, v in small_edges.keys():
                for collapsed in graph[v]:
                    if collapsed != u:
                        assert v != collapsed
                        assert not ((collapsed, v) in small_edges and (
                            v, collapsed) in small_edges)
                        if (collapsed, v) in small_edges:
                            small_edges.pop((collapsed, v))
                        elif (v, collapsed) in small_edges:
                            small_edges.pop((v, collapsed))
                        nx.set_edge_attributes(graph, {(v, collapsed): {
                            'length': math.dist(u, collapsed)}})
                        if math.dist(u, collapsed) < reduction_proximity:
                            if graph.nodes[u]['distance_to_source'] >= graph.nodes[collapsed]['distance_to_source']:
                                small_edges[(u, collapsed)] = True
                            else:
                                small_edges[(collapsed, u)] = True
                nx.contracted_edge(
                    graph, (u, v), self_loops=False, copy=False)
                small_edges.pop((u, v))
                if graph.degree(u) == 0:
                    print(f"removed node {u}, {graph[u]}")
                    graph.remove_node(u)
                break


def remove_hanging_by_graph(original_image, graph, threshold):
    changed = True
    epoch = 0
    while changed:
        new_graph = graph.copy()
        changed = False
        leafs = [node for node in graph.nodes if graph.degree(node) == 1]
        print(f"\tEpoch {epoch}")
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
            if leaf_length < threshold:
                changed = True
                for node in to_remove:
                    new_graph.remove_node(node)
        graph = new_graph
        epoch += 1
    image = original_image.copy()
    print("Finished removing hanging")
    print("Drawing result...")
    for start, end in tqdm(graph.edges()):
        cv2.line(image, start, end, (min(
            graph.nodes[start]['distance_to_source'] * 5, 255), 0, 255), thickness=1)
    cv2.namedWindow('final_result', cv2.WINDOW_NORMAL)
    cv2.imshow('final_result', image)
    cv2.imwrite('results/final_result.png', image)


def main():
    parser = argparse.ArgumentParser(description="Photo to cad")
    parser.add_argument(
        "--image", "-i", help="Path to image", default=image_path, type=str)
    parser.add_argument("--scale_factor", "-s",
                        help="Scale factor", default=4, type=int)
    parser.add_argument("--reduction_proximity", "-r",
                        help="Reduction proximity", default=2, type=int)
    parser.add_argument("--hanging_leaf_threshold", "-lt",
                        help="Hanging leaf threshold", default=80, type=int)
    parser.add_argument("--debug", "-d", help="Debug",
                        default=False, type=bool)
    args = parser.parse_args()
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
    print("Loading image...")
    image, grayscale, contours, _ = load_image(
        args.image, args.scale_factor)
    original = image.copy()
    cv2.imshow("Original", original)
    cv2.imwrite("results/original.png", original)
    cv2.drawContours(image, contours, -1, (0xff, 0, 0), 1)
    cv2.imshow('contours', image)
    cv2.imwrite("results/contours.png", image)
    print("Press any key to continue to calculation...")
    cv2.waitKey(0)
    print("calculating voronoi...")
    pv, segments = calculate_voronoi(contours, image.shape[1], image.shape[0])
    print("Finished calculating voronoi")
    print("drawing voronoi...")
    graph = draw_voronoi(image, pv, segments, args.debug)
    print("Finished drawing voronoi")
    print("Reducing graph...")
    reduce_graph(graph, args.reduction_proximity)
    print("Finished reducing graph")
    print("remove hanging...")
    remove_hanging_by_graph(original, graph, args.hanging_leaf_threshold)
    print("Finished! press Escape key to exit")
    while cv2.waitKey(0) != 27:  # Escape key
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
