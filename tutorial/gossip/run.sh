#!/bin/bash

decpy_path=../../eval # Path to eval folder
graph=64_node_fullyConnected.edges
run_path=../../eval/data # Path to the folder where the graph and config file will be copied and the results will be stored
config_file=config.ini
cp $graph $config_file $run_path

env_python=../../new_env/bin/python3    #~/miniconda3/envs/decpy/bin/python3 # Path to python executable of the environment | conda recommended
machines=4
iterations=70
test_after=50
eval_file=testing_gossip.py # decentralized driver code (run on each machine)
log_level=DEBUG # DEBUG | INFO | WARN | CRITICAL

m=`cat $(grep addresses_filepath $config_file | awk '{print $3}') | grep $(/sbin/ifconfig ens785 | grep 'inet ' | awk '{print $2}') | cut -d'"' -f2` # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=16
echo procs per machine is $procs_per_machine

log_dir=$run_path/$(date '+%Y-%m-%dT%H:%M')/machine$m
mkdir -p $log_dir

$env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $decpy_path/$graph -ta $test_after -cf $config_file -ll $log_level -wsd $log_dir
