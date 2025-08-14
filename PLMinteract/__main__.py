"""
PLM-interact: PPI prediction and mutation effect classification 
"""
import argparse
import os
import sys
from typing import Union

from .train_mlm import Train_mlm_Arguments
from .train_binary import Train_binary_Arguments
from .predict_ddp import PredictionArguments
from .inference_PPI import InferenceArguments
from .mutation_train import MutationTrainArguments
from .mutation_predict import MutationTestArguments

from . import (
    train_mlm,
    train_binary,
    predict_ddp,
    inference_PPI,
    mutation_train,
    mutation_predict
)

PLMinteractArguments = (
    Train_mlm_Arguments
    | Train_binary_Arguments
    | PredictionArguments
    | InferenceArguments
    | MutationTrainArguments
    | MutationTestArguments
)

def main():
    from . import __version__
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v", "--version", action="version", version="PLM-interact " + __version__
    )
    subparsers = parser.add_subparsers(title="PLM-interact Commands", dest="cmd")
    subparsers.required = True
    
    modules = {
        "train_mlm": train_mlm,
        "train_binary": train_binary,
        "predict_ddp": predict_ddp,
        "inference_PPI": inference_PPI,
        "mutation_train":mutation_train,
        "mutation_predict":mutation_predict,
    }

    for name, module in modules.items():
        subpara = subparsers.add_parser(name, description=module.__doc__)
        module.add_args_func(subpara)
        subpara.set_defaults(func=module.main)

    args: PLMinteractArguments = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()