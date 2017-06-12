# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 21:38:00 2016

@author: Eng.Alaa Khaled
"""

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {} # Initialize empty dictionary to store adjacent nodes
        self.distance = float("inf") # Set distance to infinity for all nodes
        self.visited = False # Mark all nodes unvisited
        self.previous = None # Predecessor nodes

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight
  
    def get_connections(self): 
        """a function to get each node adjacents/connections"""
        return self.adjacent.keys()  

    def get_id(self):
        return self.id
        
    def get_weight(self, neighbor): 
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        """set the distance/ weight of edge"""
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        """To mark/set node as visited"""
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])


class Graph:
    def __init__(self):
        self.vert_dict = {} # vertices
        self.num_vertices = 0 

    def __iter__(self):
        return iter(self.vert_dict.values())
    
    def add_vertex(self, node):
        """add a new vertex to the vertices dictionary"""
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node) # create a new vertex
        self.vert_dict[node] = new_vertex # add new vertex to the vert_dict{}
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict: # vert_dict is not empty
            return self.vert_dict[n]
        else:
            return None
             
    def add_edge(self, frm, to, cost = 0):
        """add edges from nodeX to nodeY with weightZ"""
        if frm not in self.vert_dict: 
            self.add_vertex(frm) # add frm_node if not existed
        if to not in self.vert_dict:
            self.add_vertex(to) # add to_node if not existed
        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous

def shortest(v, path):
    ''' make shortest path from v.previous, v is the Target node,
    the algorithim recursively go backward from the Target to the Start'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
    return


import heapq  
"""This module provides an implementation of the heap queue 
   algorithm, also known as the priority queue algorithm."""

def dijkstra(aGraph, start):
    print '''Dijkstra's shortest path'''
    # Set the distance for the start node to zero 
    start.set_distance(0)

    # Put tuple pair(distance,node) into the priority queue for each node in aGraph
    unvisited_queue = [(v.get_distance(),v) for v in aGraph] 
    heapq.heapify(unvisited_queue) # Transfom a list into a heap, Heaps are binary trees for which every parent 
                                   # node has a value less than or equal to any of its children.
    while len(unvisited_queue):
        # Pops a vertex with the smallest distance and mark it as the current
        uv = heapq.heappop(unvisited_queue)
        current = uv[1] 
        current.set_visited()

        #for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next) # else calculate new distence
            
            if new_dist < next.get_distance(): # if new distance to next < previously considered distance to next
                next.set_distance(new_dist) # change distance to the smallest
                next.set_previous(current) # verify that edge as the current edge to next
                print 'updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())
            else:
                print 'not updated : current = %s next = %s new_dist = %s' \
                        %(current.get_id(), next.get_id(), next.get_distance())

        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)


if __name__ == '__main__':

    g = Graph()

    g.add_vertex('a')
    g.add_vertex('b')
    g.add_vertex('c')
    g.add_vertex('d')
    g.add_vertex('e')
    g.add_vertex('f')

    g.add_edge('a', 'b', 7)  
    g.add_edge('a', 'c', 9)
    g.add_edge('a', 'f', 14)
    g.add_edge('b', 'c', 10)
    g.add_edge('b', 'd', 15)
    g.add_edge('c', 'd', 11)
    g.add_edge('c', 'f', 2)
    g.add_edge('d', 'e', 6)
    g.add_edge('e', 'f', 9)

    print 'Graph data:'
    for v in g:
        for n in v.get_connections():
            vid = v.get_id() 
            nid = n.get_id()
            print '( %s , %s, %3d)'  % ( vid, nid, v.get_weight(n)) # print node, adjacent, weight

            
    dijkstra(g, g.get_vertex('a')) # node a is the start node

    target = g.get_vertex('e') # target node is e
    path = [target.get_id()] 
    shortest(target, path)
    print 'The shortest path : %s' %(path[::-1])
 