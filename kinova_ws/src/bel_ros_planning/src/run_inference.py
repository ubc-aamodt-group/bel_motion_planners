import sys
import yaml
import os
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import pickle
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils/'))

sys.path.append(main_dir)

from utils.architectures import *

class InferenceTool(object):
    def __init__(self):
        super(InferenceTool, self).__init__()
        
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface_tutorial', anonymous=True)

        self.robot = moveit_commander.RobotCommander()

        self.scene = moveit_commander.PlanningSceneInterface()

        group_name = "arm"
        planner_id = "RRTstarkConfigDefault"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        move_group.set_planner_id(planner_id)

    def run_testing(self, configs):
        importer = fileImport()
        env_data_path = configs['data']['env_data_path']
        env_data_file = configs['data']['env_data_file']
        
        envs = importer.load_data(env_data_path, env_data_file)
        with open(env_data_path + env_data_file, 'rb') as file:
            envDict = pickle.load(file)
        
        
        path_data_path = configs['data']['path_data_path']
        path_data_file = configs['data']['path_data_file']
        pcd_data_path = configs['data']['pcd_data_path']
        
        obstacles = load_test_dataset(envs, path_data_path, pcd_data_path, path_data_file, importer, NP=100)
        
        mlp_input_size = configs['model']['mlp_input_size']
        mlp_output_size = configs['model']['mlp_output_size']
        numbits = configs['bel']['num_bits']
        dp = configs['model']['dp']
        
        encoder = Encoder()
        if(args.size == 1):
            mlp = MLP_bel_med(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
        elif(args.size == 2):
            mlp = MLP_bel_sml(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
        elif(args.size == 3):
            mlp = MLP_bel_lrg(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
        else: 
            print "Invalid model size"
            exit(1)
        
        model_path = configs['model']['model_path']
        model_file = configs['model']['mlp_model_name']
        encoder_file = configs['model']['enc_model_name']
        
        encoder.load_state_dict(torch.load(model_path + encoder_file))
        mlp.load_state_dict(torch.load(model_path + model_file))
        
        if torch.cuda.is_available():
            encoder.cuda()
            mlp.cuda()
        
        mlp.eval()
        encoder.eval()
        
        self.scene._scene_pub = rospy.Publisher('planning_scene', moveit_msgs.msg.PlanningScene, queue_size=0)
        sv = StateValidity()
        
        set_environment(self.robot, self.scene)
        
        modifier = PlanningSceneModifier(envDict['obsData'])
        
    

if __name__ == '__main__':
    with open('../../../utils/inference.yaml') as file:
        try:
            inference_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print "Error in reading yaml file"
            exit(1)
    
    inf_tool = InferenceTool()
    inf_tool.run_testing(inference_config)