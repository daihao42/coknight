import grpc,time,pickle
from utils import rpcenv_pb2
from utils import rpcenv_pb2_grpc

def rpc_connect(remote_addr):
    # connect to rpc server
    return grpc.insecure_channel(remote_addr,options=[('grpc.max_message_length',256 * 1024 * 1024)])


# send obs to gpu for inference
def inference_send(obs, reward, done, episode_step, episode_return, cut_layer, channel):
    # call rpc service
    stub = rpcenv_pb2_grpc.RPCActorInferenceStub(channel)
    response = stub.StreamingInference(rpcenv_pb2.Step(observation=pickle.dumps(obs),
                                                reward=reward,
                                                done = done,
                                                episode_step = episode_step,
                                                episode_return = episode_return,
                                                cut_layer=cut_layer))
    return response.action

# download latest model
def pull_model(actor_id, channel):
    stub = rpcenv_pb2_grpc.RPCModelUpdateStub(channel)
    response = stub.StreamingModelUpdate(rpcenv_pb2.Pull(actor_id = actor_id))
    return pickle.loads(response.parameters)

# upload trajectories
def upload_trajectory(actor_id, trajectory, channel):
    stub = rpcenv_pb2_grpc.UploadTrajectoryStub(channel)
    tr = rpcenv_pb2.Trajectory()
    tr.actor_id = actor_id
    tr.datas.extend(list(map(lambda x: rpcenv_pb2.Step(**x), trajectory))) 
    response = stub.TrajectoryUpload(tr)
    return response
