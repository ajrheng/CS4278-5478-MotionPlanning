#!/usr/bin/env python
import rospy
import numpy as np

from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from const import *
from math import *
import copy
import argparse

from base_planner import Planner
import Queue as queue

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit


class DSDAPlanner(Planner):

    def generate_plan(self):
        """TODO: FILL ME! This function generates the plan for the robot, given a goal.
        You should store the list of actions into self.action_seq.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """

        start = self.get_current_discrete_state()

        list_of_neighbors = [(1,0), (0, 1), (0, -1)] #move forward, turn left, turn right
       
        def euclidean_distance(v1, v2):
            v1 = np.array(v1)
            v2 = np.array(v2)
            return np.linalg.norm(v1-v2)

        def in_queue(q, item):
            list_of_items = [x[1] for x in q.queue]
            return item in list_of_items

        def reconstruction_path(cameFrom, action, current):
            action_seq = [action[current]]
            while current in cameFrom.keys():
                current = cameFrom[current]

                # try catch for last key that you want to skip
                try:
                    action_seq.insert(0, action[current])
                except:
                    continue

            return action_seq

        openSet = queue.PriorityQueue()

        width = int(np.ceil(self.world_width * self.resolution))
        height = int(np.ceil(self.world_height * self.resolution))
        cameFrom = {}
        action = {}
        gScore = np.array(np.ones((width, height, 4)) * np.inf)
        gScore[start] = 0
        fScore = np.array(np.ones((width, height, 4)) * np.inf)
        fScore[start] = self._d_from_goal([start[0], start[1]]) # let heuristic h be euclidean distance

        openSet.put((fScore[start], start)) # (priority, data)

        while not openSet.empty():
            
            current = openSet.get()[1] # automatically gets lowest priority (fScore) 

            if self._check_goal([current[0], current[1]]):
                self.action_seq = reconstruction_path(cameFrom, action, current)
                break
            
            for neighbor in list_of_neighbors:
                # if the neighbor doesnt cause a collision
                next_state = self.discrete_motion_predict(current[0], current[1], current[2], neighbor[0], neighbor[1])

                if next_state is not None:
                    next_state = tuple(int(s) for s in next_state)
                    tentative_gScore = gScore[current] + euclidean_distance([current[0], current[1]], [next_state[0], next_state[1]])
                    if tentative_gScore < gScore[next_state]:
                        cameFrom[next_state] = current
                        action[next_state] = neighbor
                        gScore[next_state] = tentative_gScore
                        fScore[next_state] = gScore[next_state] + self._d_from_goal([next_state[0], next_state[1]])
                        if not in_queue(openSet, next_state):
                            openSet.put((fScore[next_state], next_state)) 
       



if __name__ == "__main__":
    # TODO: You can run the code using the code below
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', type=str, default='1,8',
                        help='goal position')
    parser.add_argument('--com', type=int, default=0,
                        help="if the map is com1 map")
    args = parser.parse_args()

    try:
        goal = [int(pose) for pose in args.goal.split(',')]
    except:
        raise ValueError("Please enter correct goal format")

    if args.com:
        width = 2500
        height = 983
        resolution = 0.02
    else:
        width = 200
        height = 200
        resolution = 0.05

    # TODO: You should change this value accordingly
    inflation_ratio = 3
    planner = DSDAPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # You could replace this with other control publishers
    planner.publish_discrete_control()

    # save your action sequence
    result = np.array(planner.action_seq)
    
    filename = "DSDA_com1_" + str(goal[0]) + '_' + str(goal[1]) + ".txt"
    np.savetxt(filename, result, fmt="%.2e")
    print("wrote to ", filename)
    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')

    # spin the ros
    rospy.spin()

