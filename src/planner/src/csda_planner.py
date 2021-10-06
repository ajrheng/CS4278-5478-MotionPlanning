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
import itertools

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit


class CSDAPlanner(Planner):
    
    def generate_plan(self):
        """TODO: FILL ME! This function generates the plan for the robot, given a goal.
        You should store the list of actions into self.action_seq.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """

        def reconstruction_path(cameFrom, current):
            action = [] 
            next_x_ind, next_y_ind = int(current[0]),int(current[1])
            while (next_x_ind, next_y_ind) in cameFrom.keys():
                current = cameFrom[(next_x_ind, next_y_ind)]
                action.append(current[1])
                next_x_ind, next_y_ind = int(current[0][0]), int(current[0][1])
            return action[::-1]

        def euclidean_distance(v1, v2):
            v1 = np.array(v1)
            v2 = np.array(v2)
            return np.linalg.norm(v1-v2)
       
        scale = 6.0
        x, y, _ = self.get_current_continuous_state()
        x = int(x*scale)
        y = int(y*scale)
        start_config = [x, y, 0.0]

        w = np.arange(-np.pi/2, np.pi/2, 0.3)
        v = [0, 0.5, 1]
        list_of_neighbors = list(itertools.product(v, w))
        
        plan_x_size = int(np.ceil(self.world_width*self.resolution*scale))
        plan_y_size = int(np.ceil(self.world_height*self.resolution*scale))
        
        openSet = queue.PriorityQueue()
        cameFrom = dict()
        
        gScore = np.ones((plan_x_size, plan_y_size)) * np.inf
        gScore[x][y] = 0.0
        fScore = np.ones((plan_x_size, plan_y_size)) * np.inf
        fScore[x][y] = self._d_from_goal([x/scale, y/scale])
        
        openSet.put([fScore[x][y],[start_config, [0.0, 0.0]]])

        while not openSet.empty():

            current = openSet.get()
            curr_x, curr_y, curr_phi = current[1][0][0], current[1][0][1], current[1][0][2]
            
            if self._check_goal([curr_x/scale,curr_y/scale]):
                self.action_seq = reconstruction_path(cameFrom, (curr_x, curr_y))
                break

            for (v, theta) in list_of_neighbors:
                next_state = self.motion_predict(curr_x/scale,curr_y/scale,curr_phi,v,theta)

                if next_state is not None:
                    next_x, next_y, next_phi = next_state
                    next_x *= scale
                    next_y *= scale
                    next_node = [[next_x, next_y, next_phi], [v, theta]] # store next_x and next_y as float first
                    next_x_ind = int(next_x)
                    next_y_ind = int(next_y)
                    tentative_gScore = gScore[int(curr_x), int(curr_y)]  # no heuristic 
                    if tentative_gScore < gScore[next_x_ind, next_y_ind]:
                        tentative_fScore = tentative_gScore + self._d_from_goal([next_x/scale, next_y/scale])
                        if tentative_fScore < fScore[next_x_ind, next_y_ind]:
                            gScore[next_x_ind, next_y_ind] = tentative_gScore
                            fScore[next_x_ind, next_y_ind] = tentative_fScore 
                            cameFrom[next_x_ind, next_y_ind] = [[curr_x, curr_y, curr_phi], [v,theta]] 
                            openSet.put([fScore[next_x_ind, next_y_ind], next_node])

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
    inflation_ratio = 2
    planner = CSDAPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # You could replace this with other control publishers
    planner.publish_control()

    # save your action sequence
    result = np.array(planner.action_seq)
    filename = "CSDA_map4_" + str(goal[0]) + '_' + str(goal[1]) + ".txt"
    np.savetxt(filename, result, fmt="%.2e")
    print("wrote to ", filename)
    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')

    # spin the ros
    rospy.spin()
