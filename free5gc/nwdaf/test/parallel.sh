# Número de procesos a ejecutar
NUM_PROCESOS=2

seq 1 $NUM_PROCESOS | parallel -j $NUM_PROCESOS "echo Starting process {}; python3 nwdaf-performance-test.py t 1 > parallel-logs/output_proceso_{}.log 2>&1"