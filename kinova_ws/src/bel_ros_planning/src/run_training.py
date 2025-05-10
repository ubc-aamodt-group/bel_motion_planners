import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from tqdm import tqdm
import yaml

main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../utils/'))

sys.path.append(main_dir)

from utils.dataset import Dataset
from torch.autograd import Variable
import math
from utils.import_tools import FileImporter
import time
import sys
from utils.architectures import *
from utils.utils import to_var, joint_offset, joint_ranges, DEFAULT_STEP

def convert_soft(encode,num_bits,di):
    arr = torch.tensor(range(0,num_bits,1)).cuda()
    t = torch.matmul(encode,di)
    s = nn.Softmax(dim=1)
    t = s(t)
    t = t* arr.float()
    ts = torch.sum(t,dim=1)-20.0
    return (ts)

def decode(btbel_old,num_bits,nrange,di,joint_offset, joint_ranges):
    arr = torch.tensor(range(0,nrange,1)).cuda()
    btbel=btbel_old.view(btbel_old.size(0),6,num_bits)#*2-1
    t = torch.matmul(btbel,di.float())
    s = nn.Softmax(dim=2)
    t = s(t)
    t = t* arr.float()
    ts = torch.sum(t,dim=2)/nrange
    ts=(ts*joint_ranges)-joint_offset
    return ts

def cross_loss(btbel_old,num_bits,di):
    btbel=btbel_old.view(btbel_old.size(0),6,num_bits)#*2-1
    t = torch.matmul(btbel,di.float())   
    return t

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def get_input(i, data, targets, pc_inds, obstacles, bs):
    """
    Input:  i (int) - starting index for the batch
            data/targets/pc_inds (numpy array) - data vectors to obtain batch from
            obstacles (numpy array) - point cloud array
            bs (int) - batch size
    """
    if i+bs < len(data):
        bi = data[i:i+bs]
        bt = targets[i:i+bs]
        bpc = pc_inds[i:i+bs]
        bobs = obstacles[bpc]
    else:
        bi = data[i:]
        bt = targets[i:]
        bpc = pc_inds[i:]
        bobs = obstacles[bpc]

    return torch.from_numpy(bi), torch.from_numpy(bt), torch.from_numpy(bobs)


def main(config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    importer = FileImporter()
    env_data_path = config['env_data_path']
    path_data_path = config['path_data_path']
    pcd_data_path = config['pointcloud_data_path']
    num_paths = config['training']['num_paths']
    batch_size = config['training']['batch_size']

    envs = importer.environments_import(env_data_path + config["data"]["env_data_file"])
    params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 6}
    print("Loading obstacle data...\n")
    basedataset="bel_dataset_"+str(config["bel"]["numbits"])+"/"
    # This is to load already pickled dataset
    training_set = Dataset(envs, path_data_path, pcd_data_path, 
                           config["data"]["path_data_file"], 
                           importer, NP=num_paths,
                           filename=config["training"]["filename"]+".pkl",
                           basedataset=basedataset,
                           num_bits=config["bel"]["numbits"],
                           code=config["bel"]["code"],
                           nrange=config["bel"]["nrange"])
    with open(basedataset+"dataset_obstacles.pkl","rb") as my_file:
        obstacles=pickle.load(my_file)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    print("Loaded dataset, targets, and pointcloud obstacle vectors: ")

    if not os.path.exists(config["model"]["model_path"]):
        os.makedirs(config["model"]["model_path"])
        
    with open(config["model"]["model_path"] + '/training_info.txt', 'wb') as file:
        file.write("Training Configuration:\n")
        file.write("=======================\n")
        for key, value in vars(config).items():
            file.write("{}: {}\n".format(key, value))

    # Build the models
    mlp_input_size = config['model']['mlp_input_size']
    mlp_output_size = config['model']['mlp_output_size']
    numbits = config['bel']['num_bits']
    nrange = config['bel']['nrange']
    dp = config['model']['dp']
    size = config['model']['encoder_size']
    code = config['bel']['code']
    num_epochs = config['training']['num_epochs']
    
    if(size == 1):
        mlp = MLP_bel_sml(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
    elif(size == 2):
        mlp = MLP_bel_med(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
    elif(size == 3):
        mlp = MLP_bel_lrg(mlp_input_size, mlp_output_size,num_bits=numbits,dp=dp)
    else: 
        print "Invalid model size"
        exit(1)    
    encoder = Encoder(config['model']['enc_input_size'], config['model']['enc_output_size'], num_bits=numbits, dp=dp)

    if torch.cuda.is_available():
        encoder.cuda()
        mlp.cuda()
        
    # Loss and Optimizer
    criterion = nn.MSELoss()
    critCE=torch.nn.CrossEntropyLoss(reduction="sum").cuda()
    params = list(encoder.parameters())+list(mlp.parameters())
    optimizer = torch.optim.Adagrad(params, lr=config["training"]["learning_rate"])
    total_loss = []
    epoch = 1

    sm = 90  # start saving models after 100 epochs
    print("bel"+code+"_"+str(nrange)+"_tensor.pkl")
    di=pickle.load(open("bel"+code+"_"+str(nrange)+"_tensor.pkl","rb"))
    di=torch.transpose(di,0,1).cuda()
    print("Starting epochs...\n")
    len_dataset=training_generator.__len__()
    for epoch in range(num_epochs):
        start = time.time()
        print("epoch" + str(epoch))
        avg_loss = 0
        avg_mse_loss =0 
        cc=0
        for bi,bt,btbel,pc_inds,ct in tqdm(training_generator):
            cc+=1
            encoder.zero_grad()
            mlp.zero_grad()
            bobs = obstacles[pc_inds[:,0]]
            bi = to_var(bi)
            bt = to_var(bt)
            bobs = to_var(bobs)
            # forward pass through encoder
            h = encoder(bobs)
            # concatenate encoder output with dataset input
            inp = torch.cat((bi, h), dim=1)

            # forward pass through mlp
            bo = mlp(inp)
            if True:
                #bits, range
                cross_predicted=cross_loss(bo.cuda(),numbits,nrange,di,joint_offset)
                cross_target = ct.long().cuda()
                loss=critCE(cross_predicted.permute(0,2,1),cross_target) 
            btbel.resize_(btbel.size(0),int(6*numbits))
            
            decoded_bo=decode(bo.cuda(),numbits,nrange,di,joint_offset,joint_ranges)
            mse_loss=criterion(decoded_bo,bt)
            avg_mse_loss = avg_mse_loss+mse_loss.data
            avg_loss = avg_loss+loss.data
            
            loss.backward()
            optimizer.step()

        print("--average loss:")
        print(avg_loss/(len_dataset/batch_size))
        print("--average MSE loss:")
        print(avg_mse_loss/(len_dataset/batch_size))
        total_loss.append(avg_loss/(len_dataset/batch_size))
        # Save the models
        trained_model_path = config["model"]["model_path"]
        if epoch == sm or epoch == num_epochs-1:
            print("\nSaving model\n")
            print(trained_model_path)
            print("time: " + str(time.time() - start))
            torch.save(encoder.state_dict(), os.path.join(
                trained_model_path, 'cae_encoder_'+str(epoch)+'.pkl'))
            torch.save(total_loss, os.path.join(trained_model_path, 'total_loss_'+str(epoch)+'.dat'))

            model_path = 'mlp_PReLU_ae_dd'+str(epoch)+'.pkl'
            torch.save(mlp.state_dict(), os.path.join(
                trained_model_path, model_path))
            if (epoch != 1):
                sm = sm+10  # save model after every 50 epochs from 100 epoch ownwards

if __name__ == "__main__":
    with open('./utils/inference.yaml') as file:
        try:
            inference_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print "Error reading yaml file"
            exit(1)
    
    main(inference_config)
