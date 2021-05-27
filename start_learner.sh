python learner.py --batch_size 8 --num_buffers 101 --env SpaceInvadersNoFrameskip-v4 --unroll_length 64 --mode train  --num_actors 100  \
    --learning_rate 0.0004 \
    --epsilon 0.01 \
    --entropy_cost 0.01 \
    --total_steps 2100000 \
    --remark "cut_at_layer_2_actor_100_SpaceInvaders_async_remote"
