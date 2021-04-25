from utils import rpcenv
import pickle


if __name__ == '__main__':
    channel = rpcenv.rpc_connect("192.168.1.154:50051")
    rs = rpcenv.inference_send(obs=[[1]],reward=1,done=False,episode_step = 1,episode_return=1.0,cut_layer=2,channel=channel)
    print(rs)
    rs2 = rpcenv.pull_model(1, channel)
    print(rs2)
    rs3 = rpcenv.upload_trajectory(1,[{"observation":pickle.dumps([[1]])
                                                ,"reward":11
                                                ,"done":False
                                                ,"episode_step":1
                                                ,"episode_return":1
                                                ,"cut_layer":3},
                                            {"observation":pickle.dumps([[1]])
                                                ,"reward":10
                                                ,"done":True
                                                ,"episode_step":1
                                                ,"episode_return":1
                                                ,"cut_layer":4}],channel)
    print(rs3)
