# -*- coding: utf-8 -*-

"""
description: MySQL Database Configurations
"""

instance_config = {
    'mysql1': {
        'host': '192.168.0.11',
        'user': 'root',
        'passwd': '12345678',
        'port': 3306,
        'database': 'data',
        'memory': 34359738368
    },
    'mysql2': {
        'host': '192.168.0.15',
        'user': 'root',
        'passwd': '12345678',
        'port': 3306,
        'database': 'data',
        'memory': 34359738368
    },
    'mysql-test': {
        'host': '127.0.0.1',
        'user': 'root',
        'passwd': '12345678',
        'port': 3306,
        'database': 'sysbench_test',
        'memory': 34359738368
    },
    'mysql3': {
        'host': '8.131.229.55',
        'user': 'dbmind',
        'passwd': 'DBMINDdbmind2020',
        'port': 3306,
        'database': 'sysbench_test',
        'memory': 34359738368
    }
}



