# add input monitor
#sudo iptables -A INPUT -p tcp --dport 50051

# add output monitor
#sudo iptables -A OUTPUT -p tcp --sport 50051

# watch monitor
#sudo iptables -L -v -n -x

# reset input monitor
#sudo iptable -Z INPUT

# reset output monitor
#sudo iptable -Z OUTPUT

# remove input monitor
#sudo iptables -D INPUT -p tcp --dport 50051

# remove output monitor
#sudo iptables -D OUTPUT -p tcp --sport 50051

#### !!!! ####
#
#use dstat instead
# next 10 seconds, the network flow per second 
# dstat -tnf 1 10
# sth wrong with output
#dstat -nf -N enp4s0,50051 --nocolor --float --noheaders --output networkflows/$1
dstat -nf -N enp4s0,50051 --nocolor --float --noheaders > networkflows/$1
