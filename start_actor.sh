python actor.py --batch_size 8 --env SpaceInvadersNoFrameskip-v4 --server_address localhost:50051 --actor_index $1 --unroll_length 64 --disable_cuda --cut_layer=$2
#python actor.py --batch_size 8 --env MsPacmanNoFrameskip-v4 --server_address 172.20.110.104:50051 --actor_index $1 --unroll_length 64 --disable_cuda --cut_layer=$2
