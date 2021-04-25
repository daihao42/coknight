# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import time, pickle, os

import numpy as np
import atari_wrappers
import logging, typing, traceback
from utils import rpcenv

from core.ResnetModel import ResNet as Net
from core import environment
from core import file_writer
from core import prof
from core import vtrace

import torch


# yapf: disable
parser = argparse.ArgumentParser(description='Remote Environment Server')

parser.add_argument('--server_address', default="", type=str, 
                    help='Number of environment servers.')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                    help='Gym environment.')
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Batch size.")
parser.add_argument("--cut_layer", default=10, type=int, metavar="C",
                    help="Nerual network partition layer index.")
parser.add_argument("--actor_index", default=1, type=int, metavar="A",
                    help="Actor index.")
parser.add_argument("--unroll_length", default=80, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")


# yapf: enable

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

# mock env
class Env:
    def reset(self):
        print("reset called")
        return np.ones((4, 84, 84), dtype=np.uint8)

    def step(self, action):
        frame = np.zeros((4, 84, 84), dtype=np.uint8)
        return frame, 0.0, False, {}  # First three mandatory.


def create_env(flags):
    return atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(flags.env),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
        )
    )



def create_buffers(flags, obs_shape, num_actions) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        action=dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for key in buffers:
        buffers[key].append(torch.empty(**specs[key]))
    return buffers


def act(
    flags,
    actor_index: str,
    channel
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)

        flags.num_actions = gym_env.action_space.n

        flags.device = None
        if not flags.disable_cuda and torch.cuda.is_available():
            logging.info("Using CUDA.")
            flags.device = torch.device("cuda")
        else:
            logging.info("Not using CUDA.")
            flags.device = torch.device("cpu")

        model = Net(gym_env.observation_space.shape, num_actions=flags.num_actions
                    ).to(device=flags.device)
        model.eval()

        buffers = create_buffers(flags, gym_env.observation_space.shape, model.num_actions)

        env = environment.Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)

        while True:
            # Write old rollout end.
            for key in env_output:
                buffers[key][0][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][0][0, ...] = agent_output[key]

            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    if(flags.cut_layer < model.total_cut_layers):
                        inter_tensors, inter_T, inter_B = model(env_output, agent_state, cut_layer=flags.cut_layer)
                        agent_output, agent_state = rpcenv.inference_send(inter_tensors, agent_state, flags.cut_layer, inter_T, inter_B, env_output["reward"], channel)
                    else : 
                        agent_output, agent_state = model(env_output, agent_state)

                timings.time("model")

                env_output = env.step(agent_output["action"])

                timings.time("step")

                for key in env_output:
                    buffers[key][0][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][0][t + 1, ...] = agent_output[key]

                timings.time("write")

                # update model after a episode
                #print(env_output["done"])
                if(env_output["done"] == True):
                    parameters = rpcenv.pull_model(actor_index,channel)
                    logging.info("update model !!")
                    model.load_state_dict(parameters)
                    logging.info("update model from learner in %i", env_output["episode_step"])
                    logging.info("model return in %f", env_output["episode_return"])


            # rpc send buffers to learner
            rpcenv.upload_trajectory(actor_index,buffers,channel)


    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e

def main(flags):

    rpc_channel = rpcenv.rpc_connect(flags.server_address)
    actor_index = flags.actor_index

    act(flags,actor_index,rpc_channel)

if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
