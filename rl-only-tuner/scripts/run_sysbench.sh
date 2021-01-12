#!/usr/bin/env bash

# script_path="/usr/local/Cellar/sysbench/1.0.20/share/sysbench/"
script_path="/usr/share/sysbench/"

if [ "${1}" == "read" ]
then
    run_script=${script_path}"oltp_read_only.lua"
elif [ "${1}" == "write" ]
then
    run_script=${script_path}"oltp_write_only.lua"
else
    run_script=${script_path}"oltp_read_write.lua"
fi

sysbench ${run_script} \
        --mysql-host=$2 \
	--mysql-port=$3 \
	--mysql-user=root \
	--mysql-password=$4 \
	--mysql-db=sysbench_test \
	--db-driver=mysql \
        --mysql-storage-engine=innodb \
        --range-size=10 \
        --events=0 \
        --rand-type=uniform \
	--tables=2 \
	--table-size=10000000 \
	--report-interval=10 \
	--threads=8 \
	--time=1 \
	run >> $6
