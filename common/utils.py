import numpy as np
import inspect
import functools


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    if args.env_name == 'env_6':
        from env_6 import Environment

        # load scenario from script

        # create world
        # create multiagent environment
        env = Environment()
        args.n_agents = 2  # 需要操控的玩家个数，虽然敌人也可以控制，但是双方都学习的话需要不同的算法
        args.obs_shape = [53, 53]  # 每一维代表该agent的obs维度
        args.action_shape = [2, 2]  # 每一维代表该agent的act维度
        return env, args
    elif args.env_name == 'env_12':
        from env_12 import Environment
        env = Environment()
        args.n_agents = 4
        args.obs_shape = [209, 144, 144, 209]
        args.action_shape = [6, 4, 4, 6]
        return env, args