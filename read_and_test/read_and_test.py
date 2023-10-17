import os
import sys
import torch
from datetime import datetime
from os.path import normpath, basename
import json

from CIFAR10_simple import CIFAR10
from CIFAR10_simple import LeNet

from torch.nn import CrossEntropyLoss

from pathlib import Path

def eval(base_dir):
    """
    Reads from base_dir/models folder the saved models and evaluates them on the test set of CIFAR10.
    The filenames of the models are of the form: [node id]_model_[iteration]_iter.pt

    For each node the following dictionaries are created:
    1. test_acc: {iteration: test accuracy}
    2. test_loss: {iteration: test loss}
    3. time_test_acc: {time of creation of the model: test accuracy}
    
    The dictionaries are dumped to a json file named [node id]_testset_results.json

    """

    dataset = CIFAR10()
    model = LeNet()
    loss = CrossEntropyLoss()

    test_eval = {}

    dirs = os.listdir(base_dir)
    print(dirs)
    for dir in dirs:
        dir_path = Path(os.path.join(base_dir, dir))
        if not dir_path.is_dir():
            continue

        model_dir = Path(os.path.join(dir_path, "models"))
        files = os.listdir(model_dir)
        files = [f for f in files if f.endswith(".pt")]

        for file in files:
            print("Now reading ", file)

            model.load_state_dict(torch.load(os.path.join(model_dir, file)))
            model.eval()
            ta, tl = dataset.test(model, loss)

            file_name = file.split('_')
            node_id = file_name[0]
            iteration = file_name[2]

            if node_id not in test_eval:
                test_eval[node_id] = {"test_acc": {}, "test_loss": {}, "time_test_acc": {}}

            test_eval[node_id]["test_acc"][iteration] = ta
            test_eval[node_id]["test_loss"][iteration] = tl

            creation_time = (datetime.fromtimestamp(os.path.getctime(os.path.join(model_dir, file)))).strftime('%Y-%m-%d %H:%M:%S.%f')
            test_eval[node_id]["time_test_acc"][creation_time] = ta


    #dump results to file

    exp_datetime = basename(normpath(base_dir))

    os.makedirs(os.path.join(".", "evaluation_of_{}".format(exp_datetime)))
    for node in test_eval:
        with open(
            os.path.join("./evaluation_of_{}/".format(exp_datetime), "{}_testset_results.json".format(node)), "w"
        ) as of:
            json.dump(test_eval[node], of)



if __name__ == "__main__":
    assert len(sys.argv) == 2
    # The args are:
    # 1: path to the folder of the experiment

    eval(sys.argv[1])

