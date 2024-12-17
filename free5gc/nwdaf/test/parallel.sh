# process num
NUM_PROCESS=$1
TEST_TYPE=$2
TEST_NUM=$3

seq 1 $NUM_PROCESS | parallel -j $NUM_PROCESS "echo Starting process {}; python3 nwdaf-performance-test.py $TEST_TYPE $TEST_NUM > parallel-logs/output_proceso_{}.log 2>&1"