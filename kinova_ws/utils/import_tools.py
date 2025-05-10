import torch
import torch.utils.data as data
import os
import pickle
import numpy as np
# import nltk
from PIL import Image
import os.path
import random
from torch.autograd import Variable
import torch.nn as nn
import math
from pypcd import *
import fnmatch
import rospy
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse


class FileImporter():
	def pointcloud_import_array(self, pcd_fname, min_length_array):
		"""
		Import the pointcloud data from a file, and leave it in a (3xN) array 

		Input: 	pcd_fname (string) - filepath of file with pcd data
				min_length_array (int) - index to chop all rows of the pointcloud data to, based on environment with minimum number of points

		Return: obs_pc (numpy array) - array of pointcloud data (3XN) [x, y, z]
		"""
		pc = PointCloud.from_path(pcd_fname)

		# flatten into vector
		# obs_pc = np.zeros((3, pc.pc_data['x'].shape[0]))
		obs_pc = np.zeros((3, min_length_array))
		obs_pc[0] = pc.pc_data['x'][:min_length_array]
		obs_pc[1] = pc.pc_data['y'][:min_length_array]
		obs_pc[2] = pc.pc_data['z'][:min_length_array]

		return obs_pc

    def paths_import_all(self, path_fname):
		"""
		Import all paths from a file with paths from all environments, ofand return into a single a dictionary
		keyed by environment name. Unscramle the path data and normalize it in the process

		Input: path_fname (string)

		Return: unscrambled_dict (dictionary)
		"""
		with open (path_fname, "rb") as paths_f:
			paths_dict = pickle.load(paths_f)

		# keyed by environment name
		unscrambled_dict = {}
		for key in paths_dict.keys():
			unscrambled_dict[key] = self.moveit_unscramble(paths_dict[key])

		return unscrambled_dict

	def paths_import_single(self, path_fname, env_name, single_env=False):
		"""
		Import the paths from a single environment. File to load from may contain data from many environments, or
		just a single one, indicated by the single_env (True/False) flag

		Input: 	path_fname (string) - filepath to file with path data
				env_name (string) - name of environment to get path data for
				single_env (bool) - flag of whether path_fname indicates a file with data from multiple environments or a single env

		Return: env_paths (list) - list of numpy arrays
		"""
		if not single_env:
			with open (path_fname, "rb") as paths_f:
				paths_dict = pickle.load(paths_f)

			# for non single environment, need to use the environment name as a dictionary key to get the right path list
			env_paths = paths_dict[env_name]#self.moveit_unscramble(paths_dict[env_name])
			return env_paths

		else:
			with open (path_fname, "rb") as paths_f:
				paths_list = pickle.load(paths_f)

			env_paths = paths_list#self.moveit_unscramble(paths_list)
			return env_paths 

	def pointcloud_import(self, pcd_fname):
		"""Import the pointcloud data from a file, and flatten it into a vector

		Input: 	pcd_fname (string) - filepath of file with pcd data
		
		Return: obs_pc (numpy array) - array of pointcloud data (1X(3N))
		"""
		pc = PointCloud.from_path(pcd_fname)

		# flatten into vector
		temp = np.zeros((3, pc.pc_data['x'].shape[0]))
		temp[0] = pc.pc_data['x']
		temp[1] = pc.pc_data['y']
		temp[2] = pc.pc_data['z']

		obs_pc = temp.flatten('F') #flattened column wise, [x0, y0, z0, x1, y1, z1, x2, y2, ...]

		return obs_pc

	def pointcloud_length_check(self, pcd_fname):
		"""
		Get number of points in the pointcloud file pcd_fname
		"""
		pc = self.pointcloud_import(pcd_fname)
		return pc.shape[0]

	def environments_import(self, envs_fname):
		"""
		Import environments from files with description of where obstacles reside, dictionary keyed by 'poses' and 'obsData'. 
		This function uses the poses key, which has the positions of all the environment obstacles

		Input: envs_fname (string) - filepath of file with environment data
		Return: env_names (list) - list of strings, based on keys of dictionary in envs['poses']
		"""
		with open (envs_fname, "rb") as env_f:
			envs = pickle.load(env_f)

		env_names = envs['poses'].keys() # also has obstacle meta data
		return env_names

    def load_test_dataset(self, env_names, data_path, pcd_path, path_data_file, NP=100, min_length=5351*3):
        """
        Load dataset for end to end encoder+planner testing, which will return obstacle point clouds, paths, and path lengths

        Input:	env_names (list) - list of string names of environments to load
                data_path (string) - path to directory with path data files
                pcd_path (string) - path to directory with point cloud data files
                importer (fileImport) - object from lib to help with import functions
                NP (int) - number of paths to import from each file (should by 1000, but some files ended up with less during data generation)
                min_length (int) - known number of points in point cloud with minimum number of points (None if not known)

        Return: obstacles (numpy array) - array of pointcloud data for each environment
                paths_new (numpy array) - numpy array with test paths
                path_lenghts_new (numpy array) - numpy array with lengths of each path, to know where goal index is (the rest are padded with zeros)
        """
        N = len(env_names)
        obstacles = self.load_obstacles(env_names, pcd_path)

        ### obtain path length data ###
        # paths_file = 'trainEnvironments_testPaths_GoalsCorrect_RRTSTAR_trainEnv_4.pkl'
        paths_file = path_data_file
        print("LOADING FROM: ")
        print(paths_file)
        # calculating length of the longest trajectory
        max_length = 0
        path_lengths = np.zeros((N, NP), dtype=np.int64)
        for i, env in enumerate(env_names):
            env_paths = self.paths_import_single(
                path_fname=data_path + paths_file, env_name=env, single_env=False)
            print("env len: " + str(len(env_paths)))
            print("i: " + str(i))
            print("env name: " + env)
            env_paths = env_paths['paths']
            for j in range(0, NP):  # for j in num_paths:
                path_lengths[i][j] = len(env_paths[j])
                if len(env_paths[j]) > max_length:
                    max_length = len(env_paths[j])


        print("Obtained max path length: \n")
        print(max_length)

        ### obtain path data ###

        paths = np.zeros((N, NP, max_length, 6), dtype=np.float32)
        for i, env in enumerate(env_names):
            env_paths = self.paths_import_single(
                path_fname=data_path+paths_file, env_name=env, single_env=False)
            env_paths = env_paths['paths']
            for j in range(0, NP):
                paths[i][j][:len(env_paths[j])] = env_paths[j]

        print("Obtained paths,for envs: ")
        print(len(paths))
        print("Path matrix shape: ")
        print(paths.shape)
        print("\n")

        ### create dataset and targets ###

        # clean up paths
        paths_new = paths[:, :, 1:, :]
        path_lengths_new = path_lengths - 1

        return obstacles, paths_new, path_lengths_new
        
    def load_obstacles(self, env_names, pcd_path, minlength=(5351*3)):
        """
        Load point cloud dataset into array of obstacle pointclouds, which will be entered as input to the encoder NN, but first normalizing all the data based on mean and norm

        Input: 	env_names (list) - list of strings with names of environments to import
                pcd_data_path (string) - filepath to file with environment representation
                importer (fileImport) - object from utility library to help with importing different data
                min_length (int) - if known in advance, number of flattened points in the shortest obstacle point cloud vector

        Return: obstacles (numpy array) - array of obstacle point clouds, with different rows for different environments
                                            and different columns for points
        """
        # get file names, just grabbing first one available (sometimes there's multiple)
        fnames = []

        print("Searching for file names...")
        for i, env in enumerate(env_names):
            # hacky reordering so that we don't load the last .pcd file which is always corrupt
            # sort by the time step on the back, which helps us obtain the earliest possible
            for file in sorted(os.listdir(pcd_path), key=lambda x: int(x.split('Env_')[1].split('_')[1][:-4])):
                if (fnmatch.fnmatch(file, env+"*")):
                    fnames.append(file)
                    break

        if min_length is None: # compute minimum length for dataset will take twice as long if necessary
            min_length = 1e6 # large to start
            for i, fname in enumerate(fnames):
                length = self.pointcloud_length_check(pcd_fname=pcd_path + fname)
                if (length < min_length):
                    min_length = length

        print("Loading files, minimum point cloud obstacle length: ")
        print(min_length)
        N = len(fnames)

        # make empty array of known length, and use import tool to fill with obstacle pointcloud data
        min_length_array = min_length//3
        obstacles_array = np.zeros((3, min_length_array, N), dtype=np.float32)
        for i, fname in enumerate(fnames):
            data = self.pointcloud_import_array(pcd_path + fname, min_length_array) #using array version of import, and will flatten manually after normalization
            obstacles_array[:, :, i] = data

        # compute mean and std of each environment
        means = np.mean(obstacles_array, axis=1)
        stds = np.std(obstacles_array, axis=1)
        norms = np.linalg.norm(obstacles_array, axis=1)

        # compute mean and std of means and stds
        mean_overall = np.expand_dims(np.mean(means, axis=1), axis=1)
        std_overall = np.expand_dims(np.std(stds, axis=1), axis=1)
        norm_overall = np.expand_dims(np.mean(norms, axis=1), axis=1)

        print("mean: ")
        print(mean_overall)
        print("std: ")
        print(std_overall)
        print("norm: ")
        print(norm_overall)

        # normalize data based on mean and overall norm, and then flatten into vector
        obstacles=np.zeros((N,min_length),dtype=np.float32)
        for i in range(obstacles_array.shape[2]):
            temp_arr = (obstacles_array[:, :, i] - mean_overall)
            temp_arr = np.divide(temp_arr, norm_overall)
            obstacles[i] = temp_arr.flatten('F')

        return obstacles

DEFAULT_SV_SERVICE = "/check_state_validity"

class StateValidity():
    def __init__(self):
        rospy.loginfo("Initializing stateValidity class")
        self.sv_srv = rospy.ServiceProxy(DEFAULT_SV_SERVICE, GetStateValidity)
        rospy.loginfo("Connecting to State Validity service")
        rospy.wait_for_service("check_state_validity")
        rospy.loginfo("Reached this point")

        if rospy.has_param('/play_motion/approach_planner/planning_groups'):
            list_planning_groups = rospy.get_param('/play_motion/approach_planner/planning_groups')
        else:
            rospy.logwarn("Param '/play_motion/approach_planner/planning_groups' not set. We can't guess controllers")
        rospy.loginfo("Ready for making Validity calls")


    def close_SV(self):
        self.sv_srv.close()


    def getStateValidity(self, robot_state, group_name='both_arms_torso', constraints=None, print_depth=False):
        """Given a RobotState and a group name and an optional Constraints
        return the validity of the State"""
        gsvr = GetStateValidityRequest()
        gsvr.robot_state = robot_state
        gsvr.group_name = group_name
        if constraints is not None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)

        if not result.valid:
            contact_depths = [contact.depth for contact in result.contacts]
            max_depth = max(contact_depths)
            if max_depth < 0.0001:
                return True
            else:
                return False
        return result.valid