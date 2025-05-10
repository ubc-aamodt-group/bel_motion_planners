import sys
import yaml
import os
import rospy
import moveit_commander
import moveit_msgs.msg
import pickle
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils/'))

sys.path.append(main_dir)

from utils.architectures import *
from utils.import_tools import FileImporter, StateValidity
from utils.planning_scene_editor import *
from utils.utils import *

DEFAULT_STEP = 0.05

class InferenceTool(object):
    def __init__(self):
        super(InferenceTool, self).__init__()
        
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('bel_ros', anonymous=True)

        self.robot = moveit_commander.RobotCommander()

        self.scene = moveit_commander.PlanningSceneInterface()

        group_name = "arm"
        planner_id = "RRTstarkConfigDefault"
        move_group = moveit_commander.MoveGroupCommander(group_name)
        move_group.set_planner_id(planner_id)
        
        self.move_group = move_group

    def run_testing(self, configs):
        importer = FileImporter()
        self.configs = configs
        
        env_data_path = configs['data']['env_data_path']
        env_data_file = configs['data']['env_data_file']
        envs = importer.environments_import(env_data_path, env_data_file)
        with open(env_data_path + env_data_file, 'rb') as file:
            envDict = pickle.load(file)
        
        
        path_data_path = configs['data']['path_data_path']
        path_data_file = configs['data']['path_data_file']
        pcd_data_path = configs['data']['pcd_data_path']
        
        obstacles, paths, path_lengths = importer.load_test_dataset(envs, path_data_path, pcd_data_path, path_data_file, importer, NP=100)
        
        mlp_input_size = configs['model']['mlp_input_size']
        mlp_output_size = configs['model']['mlp_output_size']
        numbits = configs['bel']['num_bits']
        nrange = configs['bel']['nrange']
        dp = configs['model']['dp']
        
        encoder = Encoder()
        size = configs['model']['encoder_size']
        if(size == 1):
            self.mlp = MLP_bel_sml(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
        elif(size == 2):
            self.mlp = MLP_bel_med(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
        elif(size == 3):
            self.mlp = MLP_bel_lrg(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
        else: 
            print "Invalid model size"
            exit(1)
        # PlanningSceneModReady
        model_path = configs['model']['model_path']
        model_file = configs['model']['mlp_model_name']
        encoder_file = configs['model']['enc_model_name']
        
        encoder.load_state_dict(torch.load(model_path + encoder_file))
        self.mlp.load_state_dict(torch.load(model_path + model_file))
        
        if torch.cuda.is_available():
            encoder.cuda()
            self.mlp.cuda()
        
        self.mlp.eval()
        encoder.eval()
        
        self.scene._scene_pub = rospy.Publisher('planning_scene', moveit_msgs.msg.PlanningScene, queue_size=0)
        global sv 
        global filler_robot_state
        global rs_man
        sv = StateValidity()
        
        set_environment(self.robot, self.scene)
        
        master_modifier = ShelfSceneModifier()
        self.scene_modifier = PlanningSceneModifier(envDict['obsData'])
        self.scene_modifier.setup_scene(self.scene, self.robot, self.move_group)    
        
        rs_man = RobotState()
        robot_state = self.robot.get_current_state()
        rs_man.joint_state.name = robot_state.joint_state.name
        filler_robot_state = list(robot_state.joint_state.position)
        
        dof = 6
        tp = 0
        fp = 0
        neural_paths = {}

        goal_collision = []
        di = pickle.load(open(configs["model"]["model_path"]))
        di=pickle.load(open("./bel"+configs['bel']["code"]+"_"+str(numbits)+"_tensor.pkl","rb"))
        di=torch.transpose(di,0,1).cuda()

        # experiment_name = args.model_path.split('models/')[1] + "test_"
        experiment_name = model_path.split('models/')[1] + configs['results']['experiment_name']
        good_paths_path = configs['results']['good_path_sample_path'] + '/' + experiment_name
        bad_paths_path = configs['results']['bad_path_sample_path'] + '/' + experiment_name
        
        if not os.path.exists(good_paths_path):
            os.makedirs(good_paths_path)
        if not os.path.exists(bad_paths_path):
            os.makedirs(bad_paths_path)

        with open(good_paths_path + '/testing_info.txt', 'wb') as file:
            file.write("Training Configuration:\n")
            file.write("=======================\n")
            for key, value in vars(configs).items():
                file.write("{}: {}\n".format(key, value))
        with open(bad_paths_path + '/testing_info.txt', 'wb') as file:
            file.write("Training Configuration:\n")
            file.write("=======================\n")
            for key, value in vars(configs).items():
                file.write("{}: {}\n".format(key, value))
        with open("bel_paths_"+configs.experiment_name+".pkl", "wb") as f:
            x = []
            pickle.dump(x, f)

        for i, env_name in enumerate(envs):            
            col_env = []
            tp_env = 0
            fp_env = 0
            neural_paths[env_name] = []
            env_inference = []

            if not os.path.exists(good_paths_path + '/' + env_name):
                os.makedirs(good_paths_path + '/' + env_name)
            if not os.path.exists(bad_paths_path + '/' + env_name):
                os.makedirs(bad_paths_path + '/' + env_name)

            print("ENVIRONMENT: " + env_name)

            self.delete_obstacles()
            new_pose = envDict['poses'][env_name]
            self.sceneModifier.permute_obstacles(new_pose)

            for j in range(0,path_lengths.shape[1]):
                print ("step: i="+str(i)+" j="+str(j))
                print("fp: " + str(fp_env))
                print("tp: " + str(tp_env))

                obs = torch.from_numpy(obstacles[i])
                en_inp=to_var(obs)
                h=encoder(en_inp)

                if path_lengths[i][j]>0:
                    inference = 0

                    print("path length: "+ str(path_lengths[i][j]))
                    start=np.zeros(dof,dtype=np.float32)
                    goal=np.zeros(dof,dtype=np.float32)
                    for l in range(0,dof):
                        start[l]=paths[i][j][0][l]
                        goal[l]=paths[i][j][path_lengths[i][j]-1][l]

                    if (IsInCollision(goal)):
                        print("GOAL IN COLLISION --- BREAKING")
                        goal_collision.append(j)
                        continue

                    start1=torch.from_numpy(start)
                    start2=torch.from_numpy(goal)

                    path1=[]
                    path1.append(start1)
                    path2=[]
                    path2.append(start2)
                    path=[]
                    target_reached=0
                    step=0
                    tree=0
                    step_sz = DEFAULT_STEP
                    while target_reached==0 and step<50:
                        step=step+1
                        if tree==0:
                            inp1=torch.cat((start1,start2,h.data.cpu()))
                            inp1=to_var(inp1)
                            start1t=self.mlp(inp1)
                            inference += 1
                            start1=decode(start1t,numbits,nrange,di)
                            start1=start1.data.cpu()
                            path1.append(start1)
                            tree=1
                        else:
                            inp2=torch.cat((start2,start1,h.data.cpu()))
                            inp2=to_var(inp2)
                            start2t=self.mlp(inp2)
                            inference += 1
                            start2=decode(start2t,numbits,nrange,di)
                            start2=start2.data.cpu()
                            path2.append(start2)
                            tree=0
                        target_reached=steerTo(start1,start2, IsInCollision)

                    tp=tp+1
                    tp_env=tp_env+1
                    print("planning done",target_reached)
                    if (step > 50 or not target_reached):
                        self.save_feasible_path(path, bad_paths_path + '/' + env_name + '/bp_' + str(j))
                    if target_reached==1:
                        for p1 in range(0,len(path1)):
                            path.append(path1[p1])
                        for p2 in range(len(path2)-1,-1,-1):
                            path.append(path2[p2])

                        indicator=feasibility_check(path, IsInCollision, step_sz=0.01, print_depth=True)

                        path = lvc(path, IsInCollision, step_sz=step_sz)
                        indicator=feasibility_check(path, IsInCollision, step_sz=0.01, print_depth=True)
                        if indicator==1:
                            col_env.append(counter)
                            env_inference.append(inference)
                            fp=fp+1
                            fp_env=fp_env+1
                            neural_paths[env_name].append(path)
                            self.save_feasible_path(path, good_paths_path + '/' + env_name + '/fp_' + str(j))
                            print("---path found---")
                            print("length: " + str(len(path)))
                            print("collisions: " + str(counter))
                            print("inferences: " + str(inference))
                        else:
                            print("Replanning...")
                            sp=0
                            indicator=0
                            step_sz = DEFAULT_STEP
                            while indicator==0 and sp<10 and path !=0:
                                print(sp)

                                # adaptive step size on replanning attempts
                                if (sp == 1):
                                    step_sz = 0.04
                                elif (sp == 2):
                                    step_sz = 0.03
                                elif (sp > 2):
                                    step_sz = 0.02

                                sp=sp+1
                                g=torch.from_numpy(paths[i][j][path_lengths[i][j]-1])
                                print("replanning")
                                path, inf = self.replan_path(path, g, obs)
                                inference += inf
                    
                                if path !=0:
                                    path=lvc(path, step_sz=step_sz)

                                    # full dense collision check
                                    indicator=feasibility_check(path, step_sz=0.01,print_depth=True)

                                    if indicator==1:
                                        col_env.append(counter)
                                        env_inference.append(inference)
                                        fp=fp+1
                                        fp_env=fp_env+1
                                        neural_paths[env_name].append(path)
                                        self.save_feasible_path(path, good_paths_path + '/' + env_name + '/fp_' + str(j))

                                        print("---path found---")
                                        print("length: " + str(len(path)))
                                        print("collisions: " + str(counter))
                                        print("inferences: " + str(inference))

                            if (sp == 10):
                                self.save_feasible_path(path, bad_paths_path + '/' + env_name + '/bp_' + str(j))

            print("total env paths: ")
            print(tp_env)
            print("feasible env paths: ")
            print(fp_env)
            print("average collision checks: ")
            print(np.mean(col_env))
            env_data = {}
            
            # for feasible paths
            env_data['tp_env'] = tp_env
            env_data['fp_env'] = fp_env
            env_data['col_env'] = col_env
            env_data['env_inference'] = env_inference
            env_data['paths'] = neural_paths[env_name]

            with open(good_paths_path + '/' + env_name + '/env_data.pkl', 'wb') as data_f:
                pickle.dump(env_data, data_f)

        print("total paths: ")
        print(tp)
        print("feasible paths: ")
        print(fp)

        with open(good_paths_path+'/neural_paths.pkl', 'wb') as good_f:
            pickle.dump(neural_paths, good_f)
            
    def replan_path(self, p, g, obs):
        code = self.configs['bel']['code']
        nrange = self.configs['bel']['nrange']
        numbits = self.configs['bel']['num_bits']
        di=pickle.load(open("./bel"+code+"_"+str(nrange)+"_tensor.pkl","rb"))
        di=torch.transpose(di,0,1).cuda()
        inference = 0
        path = []
        path.append(p[0])
        for i in range(1, len(p)-1):
            if not IsInCollision(p[i]):
                path.append(p[i])
        path.append(g)
        new_path = []
        for i in range(0, len(path)-1):
            target_reached = False

            st = path[i]
            gl = path[i+1]
            steer = self.steerTo(st, gl)
            if steer == 1:
                new_path.append(st)
                new_path.append(gl)
            else:
                itr = 0
                pA = []
                pA.append(st)
                pB = []
                pB.append(gl)
                target_reached = 0
                tree = 0
                while target_reached == 0 and itr < 50:
                    itr = itr+1
                    if tree == 0:
                        ip1 = torch.cat((st, gl, obs.data.cpu()))
                        ip1 = to_var(ip1)
                        stt = self.mlp(ip1)
                        inference += 1
                        st=decode(stt,numbits,nrange,di)
                        st = st.data.cpu()
                        pA.append(st)
                        tree = 1
                    else:
                        ip2 = torch.cat((gl, st, obs.data.cpu()))
                        ip2 = to_var(ip2)
                        glt = self.mlp(ip2)
                        inference += 1
                        gl=decode(glt,numbits,nrange,di)
                        gl = gl.data.cpu()
                        pB.append(gl)
                        tree = 0
                    target_reached = self.steerTo(st, gl)
                if target_reached == 0:
                    print("failed to replan")
                    return 0, 0
                else:
                    for p1 in range(0, len(pA)):
                        new_path.append(pA[p1])
                    for p2 in range(len(pB)-1, -1, -1):
                        new_path.append(pB[p2])
        with open("paths_"+self.configs['results']['experiment_name']+".pkl", "rb") as f:
            paths = pickle.load(f)
        
        paths.append(new_path)
        with open("paths_"+self.configs['results']['experiment_name']+".pkl", "wb") as f:
            pickle.dump(paths, f)

        return new_path, inference
    

if __name__ == '__main__':
    with open('../../../utils/inference.yaml') as file:
        try:
            inference_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print "Error reading yaml file"
            exit(1)
    
    inf_tool = InferenceTool()
    inf_tool.run_testing(inference_config)