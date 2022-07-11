from datetime import datetime
import math
import sys
import pyvoronoi
import cv2
import numpy as np
from tqdm import tqdm
import networkx as nx
import argparse

image_path = "resources/xPhys.ppm"


def count_surrounding(image, x, y, checked, pbar):
    count = 1
    to_check = [(i, j) for i in range(max(x-1, 0), min(x+2, image.shape[0]))
                for j in range(max(y-1, 0), min(y+2, image.shape[1])) if image[i][j] == 0 and not checked[i][j] and (i != x or j != y)]
    for k, l in to_check:
        checked[k][l] = True
    to_paint = [(x, y)]
    while len(to_check) > 0:
        i, j = to_check.pop()
        checked[i][j] = True
        pbar.update(1)
        to_paint.append((i, j))
        count += 1
        surrounding = [(k, l) for k in range(max(i-1, 0), min(i+2, image.shape[0]))
                       for l in range(max(j-1, 0), min(j+2, image.shape[1])) if image[k][l] == 0 and not checked[k][l]]
        for k, l in surrounding:
            checked[k][l] = True
        to_check.extend(surrounding)
    return count, to_paint


def remove_islands(image, threshold):
    checked = [[False for _ in range(image.shape[1])]
               for _ in range(image.shape[0])]
    with tqdm(total=image.shape[0] * image.shape[1]) as pbar:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] == 0 and not checked[i][j]:
                    count, to_paint = count_surrounding(
                        image, i, j, checked, pbar)
                    if count <= threshold:
                        for x, y in to_paint:
                            image[x][y] = 255
                if not checked[i][j]:
                    pbar.update(1)
                    checked[i][j] = True
    return image


def load_image(path):
    # Get image
    image = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
    return image


def preprocess_image(image, scale_factor, islands_threshold):
    print("\tRemoving islands")
    image = remove_islands(image, islands_threshold)
    image = cv2.resize(
        image, (image.shape[1] * scale_factor, image.shape[0] * scale_factor), interpolation=cv2.INTER_NEAREST_EXACT)

    # Convert to colored
    colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Canny edges
    edged = cv2.Canny(image, 30, 200)

    # Find contours
    print("\tFinding contours")
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
    return colored, image, contours, hierarchy


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
        image = image.copy()
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
            incident_segment = segments[cell.site]
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
                                        start_vertex, distance_to_source=distance_to_edge(
                                            start_vertex, incident_segment[0], incident_segment[1]),
                                        incident_segment=incident_segment, weight=1)
                                    graph.add_node(
                                        end_vertex, distance_to_source=distance_to_edge(
                                            end_vertex, incident_segment[0], incident_segment[1]),
                                        incident_segment=incident_segment, weight=1)
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
                                                    start_vertex, distance_to_source=distance_to_edge(
                                                        start_vertex, incident_segment[0], incident_segment[1]),
                                                    incident_segment=incident_segment, weight=1)
                                                graph.add_node(
                                                    end_vertex, distance_to_source=distance_to_edge(
                                                        end_vertex, incident_segment[0], incident_segment[1]),
                                                    incident_segment=incident_segment, weight=1)
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


def remove_hanging_by_graph(graph, threshold):
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
                    if node in new_graph:
                        new_graph.remove_node(node)
        graph = new_graph
        epoch += 1
    return graph


def smooth_neighbors(graph, node, distance):
    changed = True
    while changed:
        changed = False
        for neighbor, data in graph[node].items():
            if data['length'] < distance:
                for collapsed in graph[neighbor]:
                    nx.set_edge_attributes(graph, {(neighbor, collapsed): {
                        'length': math.dist(node, collapsed)}})
                nx.contracted_nodes(graph, node, neighbor,
                                    self_loops=False, copy=False)
                changed = True
                break


def collapse_junctions(graph, threshold):
    junctions = [node for node in graph.nodes if graph.degree(node) > 2]
    for junction in tqdm(junctions):
        if junction not in graph.nodes:
            continue
        reset = True
        while reset:
            reset = False
            for neighbor, data in graph[junction].items():
                prev = junction
                length = data['length']
                route = []
                while graph.degree(neighbor) == 2:
                    route.append(neighbor)
                    next_neighbor, data = [
                        (node, data) for node, data in graph[neighbor].items() if node != prev][0]
                    prev = neighbor
                    neighbor = next_neighbor
                    length += data['length']
                    if length >= threshold:
                        break
                if length < threshold and graph.degree(neighbor) > 2:
                    for node in route:
                        graph.remove_node(node)
                    nx.contracted_nodes(graph, junction, neighbor,
                                        self_loops=False, copy=False)
                    midpoint = (junction[0] + neighbor[0]) / \
                        2, (junction[1] + neighbor[1]) / 2
                    nx.relabel_nodes(graph, {junction: midpoint}, copy=False)
                    incident_segment = graph.nodes[midpoint]['incident_segment']
                    nx.set_node_attributes(graph, {midpoint: {
                        'distance_to_source': distance_to_edge(midpoint, incident_segment[0], incident_segment[1])}})
                    for neighbor in graph[midpoint]:
                        nx.set_edge_attributes(graph, {(midpoint, neighbor): {
                            'length': math.dist(midpoint, neighbor)}})
                    smooth_neighbors(graph, midpoint, length)
                    junction = midpoint
                    reset = True
                    break


def draw_graph(image, graph, window_name, save_path=None):
    for start, end in tqdm(graph.edges()):
        cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (min(
            graph.nodes[start]['distance_to_source'] * 5, 255), 0, 255), thickness=1)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    if save_path is not None:
        cv2.imwrite(save_path, image)


def main():
    parser = argparse.ArgumentParser(description="Photo to cad")
    parser.add_argument(
        "--image", "-i", help="Path to image", default=image_path, type=str)
    parser.add_argument("--scale_factor", "-s",
                        help="Scale factor", default=4, type=int)
    parser.add_argument("--reduction_proximity", "-r",
                        help="Reduction proximity", default=2, type=int)
    parser.add_argument("--hanging_leaf_threshold", "-lt",
                        help="Hanging leaf threshold", default=250, type=int)
    parser.add_argument("--islands_threshold", "-it",
                        help="Islands size threshold", default=4, type=int)
    parser.add_argument("--junction_collapse_threshold", "-jt",
                        help="Junction collapse threshold", default=14, type=int)
    parser.add_argument("-d", "--debug", help="Debug", action='store_true')
    args = parser.parse_args()

    print("Loading image...")
    image = load_image(args.image)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.imshow("Original", image)
    cv2.imwrite("results/original.png", image)
    print("Press any key to continue to calculation...")
    cv2.waitKey(0)

    start_time = datetime.now()

    print("Preprocessing image...")
    image, grayscale, contours, _ = preprocess_image(
        image, args.scale_factor, args.islands_threshold)
    cv2.drawContours(image, contours, -1, (0xff, 0, 0), 1)
    cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
    cv2.imshow('contours', image)
    cv2.imwrite("results/contours.png", image)
    print("Finished preprocessing")

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
    graph = remove_hanging_by_graph(graph, args.hanging_leaf_threshold)
    print("Finished removing hanging")
    if args.debug:
        draw_graph(image.copy(), graph, 'Before collapsing junctions')
    print("collapsing junctions...")
    collapse_junctions(graph, args.junction_collapse_threshold)
    print("Finished collapsing junctions")

    print("Drawing result...")
    draw_graph(image.copy(), graph, 'final_result', "results/final_result.png")

    end_time = datetime.now()

    print(f"Finished! Calculated in {(end_time - start_time)}")
    print("press Escape key to exit")
    while cv2.waitKey(0) != 27:  # Escape key
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
