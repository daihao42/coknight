python learner.py --batch_size 8 --num_buffers 32 --env SpaceInvadersNoFrameskip-v4 --unroll_length 32 --mode train  --num_actors 3  \
    --learning_rate 0.0004 \
    --epsilon 0.01 \
    --entropy_cost 0.01 \
    --total_steps 300000 \
