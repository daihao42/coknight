#!/bin/bash
for i in $(seq 1 $1)
do
    echo ----start actor$i !----
    nohup sh ./start_actor.sh $i $2 > actor_logs/actor$i.log &
done
