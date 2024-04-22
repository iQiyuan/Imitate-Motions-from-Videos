import argparse

class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='terminal parse arguments')
        self.get_args()

    def get_args(self):
        
        self.parser.add_argument(
            "--task", type=str, default='imitate',
            help = "the task of the robot"
        )
        self.parser.add_argument(
            "--video", type=str, default='sample_video',
            help = "the video name"
        )
        self.parser.add_argument(
            "--GUI", type=bool, default=True,
            help = "whether to use GUI"
        )

        #———————————————————————————————————————————————————————————————————————————————————————————————————————
        # env_imitate arguments
        #———————————————————————————————————————————————————————————————————————————————————————————————————————

        self.parser.add_argument(
            "--is-discrete", type=bool, default=False,
            help = "whether the action space is discrete or not"
        )
        self.parser.add_argument(
            "--distance-threshold", type=float, default=0.1,
            help = "the distance threshold"
        )
        self.parser.add_argument(
            "--enable-smoothing", type=bool, default=True,
            help = "whether to enable smoothing"
        )
        self.parser.add_argument(
            "--enable-reference-model", type=bool, default=True,
            help = "whether to enable reference model in simulation"
        )

        #———————————————————————————————————————————————————————————————————————————————————————————————————————
        # robot arguments
        #———————————————————————————————————————————————————————————————————————————————————————————————————————

        self.parser.add_argument(
            "--use-simulation", type=bool, default=True,
            help = "whether to use simulation"
        )
        self.parser.add_argument(
            "--use-orientation", type=bool, default=True,
            help = "whether to use orientation"
        )

        #———————————————————————————————————————————————————————————————————————————————————————————————————————
        # ppo_imitate arguments
        #———————————————————————————————————————————————————————————————————————————————————————————————————————

        self.parser.add_argument(
            "--has-continuous-action-space", type=bool, default=True, 
            help="whether the action space is continuous_action_space"
        )
        self.parser.add_argument(
            "--max-ep-len", type=int, default=1000,
            help="the max episode length"
        )
        self.parser.add_argument(
            "--action-std", type=float, default=0.6,
            help="the standard deviation of the action"
        )
        self.parser.add_argument(
            "--max-training-timesteps", type=int, default=2e6,
            help="the max training timesteps"
        )
        self.parser.add_argument(
            "--save-model-freq", type=int, default=1e5,
            help="the save model freq"
        )
        self.parser.add_argument(
            "--action-std-decay-rate", type=float, default=0.05,
            help="the action std decay rate"
        )
        self.parser.add_argument(
            "--min-action-std", type=float, default=0.1,
            help="the min action std"
        )
        self.parser.add_argument(
            "--action-std-decay-freq", type=int, default=2.5e5,
            help="the action std decay freq"
        )
        self.parser.add_argument(
            "--K-epochs", type=int, default=30,
            help="the number of epochs"
        )
        self.parser.add_argument(
            "--eps-clip", type=float, default=0.2,
            help="the eps clip"
        )
        self.parser.add_argument(
            "--gamma", type=float, default=0.99,
            help="the gamma discount rate"
        )
        self.parser.add_argument(
            "--lr-actor", type=float, default=0.0003,
            help="the learning rate of the actor"
        )
        self.parser.add_argument(
            "--lr-critic", type=float, default=0.001,
        )
        self.parser.add_argument(
            "--random-seed", type=int, default=0,
            help="the random seed"
        )

    def parse_args(self):
        return self.parser.parse_args()