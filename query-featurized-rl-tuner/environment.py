import datetime
import subprocess
from collections import deque
import numpy as np

import pymysql
import pymysql.cursors as pycursor

import gym
from gym import spaces
from gym.utils import seeding


from sql2resource import SqlParser


class Database:
    def __init__(self):
        self.connection = pymysql.connect(host='127.0.0.1',
                                          port=4000,
                                          user='root',
                                          password='',
                                          db='INFORMATION_SCHEMA',
                                          cursorclass=pycursor.DictCursor)

        self.external_metric_num = 2  # [throughput, latency]
        self.internal_metric_num = 57  # 57 for tidb%count metrics, 110 for tidb%count AND tikv%count

        self.var_names = ["tidb_distsql_scan_concurrency", "tidb_hash_join_concurrency", "tidb_hashagg_final_concurrency", "tidb_hashagg_partial_concurrency", "tidb_index_join_batch_size", "tidb_index_lookup_concurrency", "tidb_index_lookup_join_concurrency", "tidb_index_lookup_size", "tidb_index_serial_scan_concurrency", "tidb_projection_concurrency", "tidb_window_concurrency", "tidb_init_chunk_size", "tidb_max_chunk_size", "tidb_opt_correlation_exp_factor", "tidb_opt_insubq_to_join_and_agg"]  # 16 dynamic variables

        self.knob_num = len(self.var_names)

    def close(self):
        self.connection.close()

    def fetch_internal_metrics(self):
        with self.connection.cursor() as cursor:
            sql = "SELECT sum_value FROM information_schema.metrics_summary WHERE metrics_name LIKE 'tidb%count'" #57 total metrics
            # sql = "SELECT * FROM information_schema.metrics_summary WHERE metrics_name LIKE 'tidb%count' OR metrics_name LIKE 'tikv%count" #110 total metrics
            cursor.execute(sql)
            result = cursor.fetchall()
            state_list = np.array([])
            for s in result:
                state_list = np.append(state_list, [s['sum_value']])

            return state_list

    def fetch_knob(self):
        with self.connection.cursor() as cursor:
            # @@tidb_opt_agg_push_down, tidb_opt_distinct_agg_push_down # Are both Session only variables not global
            sql = "select @@tidb_distsql_scan_concurrency, @@tidb_hash_join_concurrency, @@tidb_hashagg_final_concurrency, @@tidb_hashagg_partial_concurrency, @@tidb_index_join_batch_size, @@tidb_index_lookup_concurrency, @@tidb_index_lookup_join_concurrency, @@tidb_index_lookup_size, @@tidb_index_serial_scan_concurrency, @@tidb_projection_concurrency, @@tidb_window_concurrency, @@tidb_init_chunk_size, @@tidb_max_chunk_size, @@tidb_opt_correlation_exp_factor, @@tidb_opt_insubq_to_join_and_agg"

            cursor.execute(sql)
            result = cursor.fetchall()
            state_list = np.array([])

            i = 0
            state_list = []
            for i in range(self.knob_num):
                state_list = np.append(state_list, result[0]["@@%s" % self.var_names[i]])

            return state_list

    def change_knob_nonrestart(self, actions):
        with self.connection.cursor() as cursor:
            for i in range(self.knob_num):
                sql = 'set global %s=%d' % (self.var_names[i], actions[i])
                cursor.execute(sql)


o_low = np.array(
    [-10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000,
     -10000000000, -10000000000
     # ,            0,            0,          100,            0,            0
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000,
     -10000000000, -10000000000
     ])
o_high = np.array(
    [10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000
        , 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000
        , 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000
        , 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000
        , 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000
        , 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000
        , 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000
        , 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000, 10000000000,
     -10000000000
     # ,        100,     1000000,       100000,      100000,           1
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000
        , -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000, -10000000000,
     -10000000000, -10000000000
     ])

a_low = np.array([1, 1, 1, 1, 500, 1, 1, 500, 1, 1, 1, 1, 32, 0, 0])
a_high = np.array([32, 32, 32, 32, 100000, 32, 32, 100000, 32, 32, 32, 32, 10000, 100, 1])

# Define the environment
class Environment(gym.Env):

    def __init__(self, db, argus):

        self.db = db

        self.parser = SqlParser(cur_op=argus['cur_op'], num_event=argus['num_event'], p_r_range=argus['p_r_range'],
                                p_u_index=argus['p_u_index'], p_i=argus['p_i'], p_d=argus['p_d'])

        self.state_num = db.internal_metric_num
        self.action_num = db.knob_num

        # state_space
        self.o_low = o_low
        self.o_high = o_high

        # high = np.array([1., 1., self.max_speed])
        self.observation_space = spaces.Box(low=self.o_low, high=self.o_high, dtype=np.float32)
        self.state = db.fetch_internal_metrics()
        # print("Concatenated state:")
        # self.state = np.append(self.parser.predict_sql_resource(), self.state)
        print(self.state)

        # action_space
        # Offline
        # table_open_cache(1), max_connections(2), innodb_buffer_pool_instances(4),
        # innodb_log_files_in_group(5), innodb_log_file_size(6), innodb_purge_threads(7), innodb_read_io_threads(8)
        # innodb_write_io_threads(9),

        # Online
        # innodb_buffer_pool_size(3), max_binlog_cache_size(10), binlog_cache_size(11)
        # 1 2 3 11
        # exclude
        # innodb_file_per_table, skip_name_resolve, binlog_checksum,
        # binlog_format(dynamic, [ROW, STATEMENT, MIXED]),

        self.a_low = a_low
        self.a_high = a_high
        self.action_space = spaces.Box(low=self.a_low, high=self.a_high, dtype=np.int32)
        self.default_action = self.a_low

        self.mem = deque(maxlen=argus['maxlen_mem'])  # [throughput, latency]
        self.predicted_mem = deque(maxlen=argus['maxlen_predict_mem'])

        self.seed()
        self.start_time = datetime.datetime.now()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def preheat(self):
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_read_only.lua --threads=4 --events=0 --time=40 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph' --mysql-port=3306 --tables=5 --table-size=1000000 --range_selects=off --db-ps-mode=disable --report-interval=1 --mysql-db='sbtest' run >/home/zxh/fl_preheat 2>&1"
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_write_only.lua --threads=10 --time=40 --events=0 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph'\
        # --mysql-port=3306 --tables=10 --table-size=500000 --db-ps-mode=disable --report-interval=10 --mysql-db='sbtest_wo_2' run >/home/zxh/fl_preheat 2>&1"
        # p = subprocess.check_output(self.parser.cmd + ' cleanup', shell=True)
        p = subprocess.check_output(self.parser.cmd + ' prepare', shell=True)

        p = subprocess.check_output(self.parser.cmd + ' run', shell=True)

        p = subprocess.check_output(self.parser.cmd + ' cleanup', shell=True)
        print("Preheat Finished")

    def fetch_action(self):
        return self.db.fetch_knob()

    # new_state, reward, done,
    def step(self, u, isPredicted, iteration):
        print(u)
        u = u.astype(np.int)
        self.db.change_knob_nonrestart(u)
        # 1 run sysbench
        # primary key lookup
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_read_only.lua --threads=4 --events=0 --time=20 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph' --mysql-port=3306 --tables=5 --table-size=1000000 --range_selects=off --db-ps-mode=disable --report-interval=1 --mysql-db='sbtest' run >/home/zxh/fl1 2>&1"
        # cmd = "sysbench /home/zxh/sysbench/src/lua/oltp_write_only.lua --threads=10 --time=30 --events=0 --mysql-host=127.0.0.1 --mysql-user='root' --mysql-password='110xph'\
        # --mysql-port=3306 --tables=10 --table-size=500000 --db-ps-mode=disable --report-interval=10 --mysql-db='sbtest_wo_2' run >/home/zxh/fl1 2>&1"
        # self.parser.cmd
        # p = subprocess.check_output(self.parser.cmd + ' cleanup', shell=True)
        p = subprocess.check_output(self.parser.cmd + ' prepare', shell=True)

        p = subprocess.check_output(self.parser.cmd + ' run >fl1 2>&1', shell=True)

        p = subprocess.check_output(self.parser.cmd + ' cleanup', shell=True)
        ifs = open('fl1', 'r')
        for line in ifs.readlines():
            a = line.split()
            if len(a) > 2 and 'transactions:' == a[0]:
                throughput = float(a[2][1:])
                # print('T: '+str(throughput))
            if len(a) > 1 and 'avg:' == a[0]:
                latency = float(a[1][:2])

        # print(str(len(self.mem)+1)+"\t"+str(throughput)+"\t"+str(latency))
        cur_time = datetime.datetime.now()
        interval = (cur_time - self.start_time).seconds
        self.mem.append([throughput, latency])
        # 2 refetch state
        self._get_obs()

        # 3 cul reward(T, L)
        if len(self.mem) != 0:
            dt0 = (throughput - self.mem[0][0]) / self.mem[0][0]
            dt1 = (throughput - self.mem[len(self.mem) - 1][0]) / self.mem[len(self.mem) - 1][0]
            if dt0 >= 0:
                rt = ((1 + dt0) ** 2 - 1) * abs(1 + dt1)
            else:
                rt = -((1 - dt0) ** 2 - 1) * abs(1 - dt1)

            dl0 = -(latency - self.mem[0][1]) / self.mem[0][1]
            dl1 = -(latency - self.mem[len(self.mem) - 1][1]) / self.mem[len(self.mem) - 1][1]
            if dl0 >= 0:
                rl = ((1 + dl0) ** 2 - 1) * abs(1 + dl1)
            else:
                rl = -((1 - dl0) ** 2 - 1) * abs(1 - dl1)

        else:  # initial action
            rt = 0
            rl = 0

        reward = 6 * rl + 4 * rt
        '''
        reward = 0
        for i in range(u.shape[0]):
            tmp = u[i] / self.a_high[i]
            reward+=tmp
        '''

        '''
        print("Performance: %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))
        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])
            if len(self.predicted_mem)%10 == 0:
                print("Predict List")
                print(self.predicted_mem)
       '''

        if isPredicted:
            self.predicted_mem.append([len(self.predicted_mem), throughput, latency, reward])
            # if len(self.predicted_mem)%10 == 0:
            # print("Predict List")
            # print(self.predicted_mem)
            print("Predict %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            self.pfs = open('rw_predict_2', 'a')
            self.pfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            self.pfs.close()
        else:
            print("Random %d\t%f\t%f\t%f\t%ds" % (len(self.mem) + 1, throughput, latency, reward, interval))

            self.rfs = open('rw_random_2', 'a')
            self.rfs.write("%d\t%f\t%f\n" % (iteration, throughput, latency))
            self.rfs.close()

        return self.state, reward, False, {}

    def _get_obs(self):
        self.state = self.db.fetch_internal_metrics()
        # self.state = np.append(self.parser.predict_sql_resource(), self.state)
        return self.state
