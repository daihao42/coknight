from concurrent import futures
import time,pickle
import grpc
from utils import rpcenv_pb2,rpcenv_pb2_grpc


class ActorInferenceRpc(rpcenv_pb2_grpc.RPCActorInferenceServicer):
    def StreamingInference(self, request, context):
        print(pickle.loads(request.observation)) 
        return rpcenv_pb2.Action(action = request.cut_layer)

class ActorUpdateModelRPC(rpcenv_pb2_grpc.RPCModelUpdateServicer):
    def StreamingModelUpdate(self, request, context):
        print(request.actor_id)
        return rpcenv_pb2.Model(parameters = pickle.dumps({"a":1231}))

class ActorUploadTrajectoryRPC(rpcenv_pb2_grpc.UploadTrajectoryServicer):
    def TrajectoryUpload(self, request, context):
        print("upload",request.actor_id)
        print(pickle.loads(request.datas))
        return rpcenv_pb2.Uploaded(ack = "ok")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
        ('grpc.max_send_message_length', 64 * 1024 * 1024),
        ('grpc.max_receive_message_length', 64 * 1024 * 1024)])

    rpcenv_pb2_grpc.add_RPCActorInferenceServicer_to_server(ActorInferenceRpc(), server)
    rpcenv_pb2_grpc.add_RPCModelUpdateServicer_to_server(ActorUpdateModelRPC(), server)
    rpcenv_pb2_grpc.add_UploadTrajectoryServicer_to_server(ActorUploadTrajectoryRPC(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
