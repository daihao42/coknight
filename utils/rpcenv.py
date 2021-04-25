import grpc,time,pickle
from utils import rpcenv_pb2
from utils import rpcenv_pb2_grpc

def rpc_connect(remote_addr):
    # connect to rpc server
    return grpc.insecure_channel(remote_addr,options=[('grpc.max_message_length',1024 * 1024 * 1024),
        ('grpc.max_receive_message_length',1024 * 1024 * 1024)])


# send obs to gpu for inference
def inference_send(inter_tensors, agent_state, cut_layer, T, B, reward, channel):
    # call rpc service
    stub = rpcenv_pb2_grpc.RPCActorInferenceStub(channel)
    response = stub.StreamingInference(rpcenv_pb2.Step(inter_tensors=pickle.dumps(inter_tensors),
                                                agent_state=pickle.dumps(agent_state),
                                                cut_layer=cut_layer,
                                                T=T, B=B,
                                                reward=pickle.dumps(reward)))
    return pickle.loads(response.agent_output_state)

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
    tr.datas = pickle.dumps(trajectory)
    response = stub.TrajectoryUpload(tr)
    return response
