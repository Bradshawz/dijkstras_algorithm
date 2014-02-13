'''
CMPUT 275 Assignment 3 Part 1: Map Server

Ross Anderson
Andrew Bradshaw

Lec EB1
'''

from Graph import Graph, least_cost_path
import math
import sys
import serial

DEBUG = False

def dbprint(msg):
    if DEBUG:
        print(msg)

def load_map(filename):
    """
    Loads map data from the filename provided.
    
    Data file must be in the format:
       for Vertex:
    vertices, VERTEX_ID, latitude, longitude
       for Edge:
    edges, VERTEX_1, VERTEX_2, EDGE_NAME
    
    Returns a tuple containing (graph, position, street names, cost_function)
    """
    vertices = set()
    position = dict()
    edges = []
    street_names = dict()
    file = open(filename)
    for line in file:
        linedata = line.split(',')
        if linedata[0] == 'V':
            # VERTEX
            vertex = int(linedata[1])
            latitude = math.trunc(int(linedata[2].replace('.','')) / 10)
            longitude = math.trunc(int(linedata[3].replace('.','').replace('\n','')) / 10)
            vertices.add(vertex)
            position[vertex] = (latitude,longitude)
        elif linedata[0] == 'E':
            # EDGE
            start = int(linedata[1])
            end = int(linedata[2])
            edge_name = linedata[3].replace('\n','')
            edges.append((start, end))
            street_names[(start, end)] = edge_name
        else:
            # Something else, not sure what happened,
            # just continue to next iteration.
            # This case should never occur
            print("Line in data file was not an edge or vertex.\nLine was:", line)
    file.close()
    
    # Define the cost function
    def cost_distance(edge):
        start, end = edge[0], edge[1]
        # pythagorean theorem -> d = \|x^2 + y^2|
        x = math.fabs(position[start][0] - position[end][0])
        y = math.fabs(position[start][1] - position[end][1])
        return math.sqrt(x**2 + y**2)
    
    graph = Graph(vertices, edges)
    return (graph, position, street_names, cost_distance)

def get_input(ser):
    raw_input = ser.readline().decode('ASCII')
    values = [int(value) for value in raw_input.split()]
    start_pos, end_pos = (values[0],values[1]), (values[2],values[3])
    return (start_pos, end_pos)

def get_nearest(position, all_points):
    """
    Finds the nearest point in all_points to the lat/lon pair specified by position.
    all_points is a dictionary mapping vertex ids to lat/lon pairs
    
    >>> get_nearest((50,50), {1:(50,50), 2:(50,60)})
    1
    >>> get_nearest((10,20), {1:(0,10), 2:(10, 10)})
    2
    """
    min_distance = float('infinity')
    closest = -1
    for vertex, latlon in all_points.items():
        # pythagorean theorem -> d = \|x^2 + y^2|
        x = latlon[0] - position[0]
        y = latlon[1] - position[1]
        dist = math.sqrt(x**2 + y**2)
        if dist < min_distance:
            closest = vertex
            min_distance = dist
    return closest
        

def main():
    
    ser = serial.Serial('/dev/ttyACM0', 9600)
    graph, position, streetnames, cost = load_map("edmonton-roads-2.0.1.txt")
    
    server_running = True
    while (server_running):
        start_pos, end_pos = get_input(ser)
        dbprint(['start_pos:', start_pos, 'end_pos', end_pos])
        
        start, end = get_nearest(start_pos, position), get_nearest(end_pos, position)
        dbprint(['start:', start, 'end:', end])
        
        path = least_cost_path(graph, start, end, cost)
        dbprint(['path:', path])
        
        if len(path):
            ser.write(str(print(len(path))).encode('ASCII'))
            for el in path:
                outp = str(position[el][0])+' '+str(position[el][1])
                ser.write(outp.encode('ASCII'))
                print(outp)
        else:
            print("Path not found. Please check that the points you specified are connected.")
    

if __name__ == '__main__':
    main()
