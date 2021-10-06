#!/usr/bin/env python
from numpy.lib.function_base import percentile
import rospy
import numpy as np

from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from const import *
from math import *
import copy
import argparse

from base_planner import Planner, dump_action_table
import matplotlib.pyplot as plt

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit

class DSPAPlanner(Planner):

    pass

  
    def value_iteration(self):

        self.width = int(np.ceil(self.world_width * self.resolution))
        self.height = int(np.ceil(self.world_height * self.resolution))
        directions = 4
        discount = 0.9
        value_table = np.zeros((self.width, self.height, directions), dtype=float)
        (goal_x, goal_y) = self._get_goal_position()
        goal_reward = 1.
        value_table[goal_x, goal_y, :] = goal_reward
        mse_threshold = 1e-10

        # get all indices that are not obstacles
        # excluding goal state from this set of indices somehow helps to avoid local minmimums in VI
        ind = []
        for x in range(self.width):
            for y in range(self.height):
                for theta in [0, 1, 2, 3]:
                    if self.discrete_motion_predict(x, y, theta, 0, 0) is not None \
                        and (x, y) != (goal_x, goal_y):
                        ind.append([x, y, theta])
        ind = np.array(ind)
        
        while True:

            value_table_temp = value_table.copy()

            for indices in ind:
                curr_x, curr_y, curr_theta = indices[0], indices[1], indices[2]
                val_list = []

                for (v,w) in [(1,0), (0, 1), (0, -1)]:
                    next_state = self.discrete_motion_predict(curr_x, curr_y, curr_theta, v, w)

                    if next_state is not None:
                        # if the next action was a move forward
                        next_x, next_y, next_theta = int(next_state[0]), int(next_state[1]), int(next_state[2])

                        if abs(next_x - curr_x) == 1 or abs(next_y - curr_y) == 1:
                            value =  0.9 * value_table_temp[next_x, next_y, next_theta]
                            for [v,w] in [[np.pi/2, 1], [np.pi/2, -1]]:
                                off_position = self.discrete_motion_predict(curr_x, curr_y, curr_theta, v, w)
                                # if it was none, means collision, the value is 0, so no need to add anything
                                # if off_position is not None:
                                #     value +=  0.05 * value_table_temp[int(off_position[0]),  int(off_position[1]), int(off_position[2])]
                                if off_position is None:
                                    value *= 0.7

                        # if rotate in place 
                        else:
                            value = value_table_temp[next_x, next_y, next_theta]

                        val_list.append([value, (next_x, next_y, next_theta)])

                # take action with max V(t)
                value, (next_x, next_y, next_theta) = sorted(val_list)[-1]
                value_table[curr_x, curr_y, curr_theta] = value
                
            # since ind doesnt contain goal, we manually update it here
            # add constant to make sure it is always maximum
            for theta in [0, 1, 2, 3]:
                value_table[goal_x, goal_y, theta] += goal_reward
            
            value_table *= discount

            print("Current MSE: ", ((value_table_temp - value_table)**2).mean())
            # break when MSE is below threshold, i.e., VI converged
            if ((value_table_temp - value_table)**2).mean() < mse_threshold:
                break

        return value_table

    def generate_plan(self):

        def next_pos_to_action(curr_x, curr_y, curr_theta, next_x, next_y, next_theta):
            action_space = [
                (1, 0),
                (0, 1),
                (0, -1)
            ]

            for (v,w) in action_space:
                next_state = self.discrete_motion_predict(curr_x, curr_y, curr_theta, v, w)
                # print(v,w, next_state)
                if next_state is not None:
                    if next_state == (next_x, next_y, next_theta):
                        return (v,w)
            
            return None # couldnt transition from one state to next cause obstacle blocking in between
        

        
        value_table = self.value_iteration()

        # init action table
        self.action_table = {}
        for x in range(self.width):
            for y in range(self.height):
                for theta in [0, 1, 2, 3]:
                    key = str(x) + ',' + str(y) + ',' + str(theta)
                    self.action_table[key] = (0, 0)
        
        # get all indices that are not obstacles
        ind = []        
        for x in range(self.width):
            for y in range(self.height):
                for theta in [0, 1, 2, 3]:
                    if self.discrete_motion_predict(x, y, theta, 0, 0) is not None:
                        ind.append([x, y, theta])
        ind = np.array(ind)

        for indices in ind:
            curr_x, curr_y, curr_theta = indices[0], indices[1], indices[2]
            val_list = []

            for (v,w) in [(1,0), (0, 1), (0, -1)]:
                next_state = self.discrete_motion_predict(curr_x, curr_y, curr_theta, v, w)
                if next_state is not None:
                    next_x, next_y, next_theta = int(next_state[0]), int(next_state[1]), int(next_state[2])
                    value = value_table[next_x, next_y, next_theta]
                    val_list.append([value, (next_x, next_y, next_theta)])

            val_list = sorted(val_list)
            while True:
                best_next_x, best_next_y, best_next_theta =val_list[-1][1]

                # if the best neighboring state involves crossing an obstacle, next_pos_to_action will return none
                vw = next_pos_to_action(curr_x, curr_y, curr_theta, best_next_x, best_next_y, best_next_theta)
                if vw is not None:
                    v, w = vw[0], vw[1]
                    break
                val_list.pop()

            key = str(curr_x) + ',' + str(curr_y) + ',' + str(curr_theta)
            self.action_table[key] = (v,w)


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
    planner = DSPAPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()
        
    # You could replace this with other control publishers
    planner.publish_stochastic_control()

    # save your action sequence
    filename = "DSPA_com1_" + str(goal[0]) + '_' + str(goal[1]) + ".json"
    dump_action_table(planner.action_table, filename)
    print("wrote to ", filename)

    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')

    # spin the ros
    rospy.spin()

