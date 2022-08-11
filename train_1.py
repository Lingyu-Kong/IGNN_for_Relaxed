import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import wandb
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from gnn.ignn import IGNN
from gnn.metalayer import MLPwoLastAct
from utils.tensor_utils import to_tensor

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--memory_size", type=int, default=5000)
args = parser.parse_args()

shared_params={
    "num_atoms":60,
    "device":device,
    "batch_size":args.batch_size,
    "memory_size":args.memory_size,
    "num_epochs":args.epochs,
    "lr":args.lr
}

critic_params={
        "gnn_params":{
                "device":device,
                "num_atoms":shared_params["num_atoms"],
                "mlp_hidden_size":256,
                "mlp_layers":2,
                "latent_size":64,
                "use_layer_norm":False,
                "num_message_passing_steps":6,
                "global_reducer":"sum",
                "node_reducer":"sum",
                "dropedge_rate":0.1,
                "dropnode_rate":0.1,
                "dropout":0.1,
                "layernorm_before":False,
                "use_bn":False,
                "cycle":1,
                "node_attn":True,
                "global_attn":True,
            },
        "mlp_params":{
                "input_size":64,
                "output_sizes":[64]*2+[1],
                "use_layer_norm":False,
                "activation":nn.ReLU,
                "dropout":0.1,
                "layernorm_before":False,
                "use_bn":False,
            },
        "lr":5e-4,
        "decay_interval":200,
        "decay_rate":0.95,
        "device":shared_params["device"],
}

NUM_EPOCHS=args.epochs
NUM_ATOMS=shared_params["num_atoms"]
BATCH_SIZE=shared_params["batch_size"]
MEMORY_SIZE=shared_params["memory_size"]

class Critic(nn.Module):
    def __init__(
        self,
        gnn_params: dict,
        mlp_params: dict,
        lr: float,
        decay_interval:int,
        decay_rate:float,
        device:torch.device,
    ):
        super().__init__()
        self.gnn = IGNN(**gnn_params)
        self.mlp = MLPwoLastAct(**mlp_params)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler=StepLR(self.optimizer,decay_interval,decay_rate)
        self.device=device
        self.gnn.to(device)
        self.mlp.to(device)
        self.weight_init()
    
    def forward(self,conforms):
        """
        conforms:[batch_size,num_atoms,3]
        """
        _,_,global_attr=self.gnn(conforms)
        value=self.mlp(global_attr)
        return value

    def save_model(self,path):
        torch.save(self.state_dict(),path)
    
    def load_model(self,path):
        self.load_state_dict(torch.load(path))

    def weight_init(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)

wandb.login(key="37f3de06380e350727df28b49712f8b7fe5b14aa")
wandb.init(project="IGNN for Relaxed",entity="kly20",config=shared_params)

if __name__=="__main__":
    critic=Critic(**critic_params)
    data=np.load("rss_"+str(NUM_ATOMS)+".npz")
    conforms_memory=data["positions"]
    choices=np.random.choice(conforms_memory.shape[0],size=MEMORY_SIZE*1.2,replace=False)
    conforms_memory=conforms_memory[choices]
    energies_memory=data["energies"][choices]
    train_conforms=conforms_memory[:MEMORY_SIZE]
    train_energies=energies_memory[:MEMORY_SIZE]
    test_conforms=conforms_memory[MEMORY_SIZE:]
    test_energies=energies_memory[MEMORY_SIZE:]
    for i in range(NUM_EPOCHS): 
        start_time=time.time()
        choices=np.random.choice(train_conforms.shape[0],size=BATCH_SIZE,replace=False)
        conforms=train_conforms[choices]
        energies=train_energies[choices]
        value=critic(to_tensor(conforms))
        loss=F.mse_loss(value,to_tensor(energies))
        critic.optimizer.zero_grad()
        loss.backward()
        critic.optimizer.step()
        critic.scheduler.step()
        end_time=time.time()
        wandb.log({"train loss":loss.item()})
        if i%100==0:
            print("Epoch: {}, loss: {}, time: {}".format(i,loss.item(),end_time-start_time))
            choices=np.random.choice(test_conforms.shape[0],size=BATCH_SIZE,replace=False)
            conforms=test_conforms[choices]
            energies=test_energies[choices]
            value=critic(to_tensor(conforms))
            loss=F.mse_loss(value,to_tensor(energies))
            wandb.log({"test loss":loss.item()})
    
    ## final visualization
    for i in range(10):
        choices=np.random.choice(test_conforms.shape[0],size=BATCH_SIZE,replace=False)
        conforms=test_conforms[choices]
        energies=test_energies[choices]
        values=critic(to_tensor(conforms))
        plt.figure()
        plt.plot(energies.tolist(),label="ground truth")
        plt.plot(values.tolist(),label="prediction")
        plt.legend()
        wandb.log("compare_"+str(i),plt)