"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import importlib


def find_trainer_using_name(trainer_name):
    # Given the option --trainer [trainername],
    # the file "trainers/trainername_trainer.py"
    # will be imported.
    trainer_filename = "trainers." + trainer_name + "_trainer"
    trainerlib = importlib.import_module(trainer_filename)

    # In the file, the class called TrainerNameTrainer() will
    # be instantiated.
    trainer = None
    target_trainer_name = trainer_name.replace('_', '') + 'trainer'
    for name, cls in trainerlib.__dict__.items():
        if name.lower() == target_trainer_name.lower():
            trainer = cls

    if trainer is None:
        print("In %s.py, there should be a trainer class with class name that matches %s in lowercase." % (trainer_filename, target_trainer_name))
        exit(0)

    return trainer


def get_option_setter(trainer_name):
    trainer_class = find_trainer_using_name(trainer_name)
    return trainer_class.modify_commandline_options


def create_trainer(opt):
    trainer = find_trainer_using_name(opt.trainer)
    instance = trainer(opt)
    print("trainer [%s] was created" % (type(instance).__name__))

    return instance
