import torch
from src.train import prepare_and_train
from src.utils import device, get_shipWatcher

shipWatcher = get_shipWatcher()
shipWatcher = shipWatcher.to(device)

lr_list = [0.0005, 0.001, 0.003, 0.005]
weight_decay_list = [0.0005, 0.001, 0.005]

optimal_model = ""
train_loss = 0
accuracy = -1
state_dict = None

for lr in lr_list:
    for wd in weight_decay_list:
        output = prepare_and_train(10, lr, wd)
        if output[1] > accuracy:
            optimal_model = f"lr={lr},wd={wd}"
            train_loss = output[0]
            accuracy = output[1]
            state_dict = output[2]

print(f"Optimal Model({optimal_model}) has loss={train_loss}, accuracy={accuracy}")
torch.save(state_dict, "shipWatcher.pth")
