python learner.py --batch_size 8 --num_buffers 24 --env MsPacmanNoFrameskip-v4 --unroll_length 64 --mode train  --num_actors 6  \
    --learning_rate 0.0004 \
    --epsilon 0.01 \
    --entropy_cost 0.01 \
    --total_steps 6100000 \
    --remark "cut_at_layer_5_actor_8_MsPacman_bandwidth"
