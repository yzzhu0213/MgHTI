import argparse
import torch


def read_args():
    dim = 128
    parser = argparse.ArgumentParser()
    args = parser.parse_args().__dict__
    args["herb_input_dim"] = 646
    args["herb_output_dim"] = dim
    args["target_input_dim"] = dim
    args["target_output_dim"] = dim
    args["ingredient_input_dim"] = 78
    args["ingredient_output_dim"] = dim

    args["lr"] = 0.0001
    args["dropout"] = 0.2
    args["weight_decay"] = 0.001

    args["num_epochs"] = 200
    args["batch_size"] = 512
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"

    args["lambda1"] = 0.1
    args["lambda2"] = 0.01

    return args
