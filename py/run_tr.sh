# nohup sh run_tr.sh &

nohup python -u 002_pivot_oid.py 0 > LOG/log_002_pivot_oid.py.txt &
nohup python -u 003_pivot_oid-pb.py 0 > LOG/log_003_pivot_oid-pb.py.txt &
nohup python -u 004_pivot_oid-year.py 0 > LOG/log_004_pivot_oid-year.py.txt &

sleep 60

nohup python -u 005_pivot_oid-pb-year.py 0 > LOG/log_005_pivot_oid-pb-year.py.txt &
nohup python -u 006_pivot_highest_oid.py 0 > LOG/log_006_pivot_highest_oid.py.txt &
nohup python -u 007_pivot_lowest_oid.py 0 > LOG/log_007_pivot_lowest_oid.py.txt &
nohup python -u 008_pivot_diff_oid.py 0 > LOG/log_008_pivot_diff_oid.py.txt &
nohup python -u 009_pivot_highest_oid-pb.py 0 > LOG/log_009_pivot_highest_oid-pb.py.txt &


sleep 60


nohup python -u 010_pivot_lowest_oid-year.py 0 > LOG/log_010_pivot_lowest_oid-year.py.txt &
nohup python -u 011_pivot_fft_oid-pb.py 0 > LOG/log_011_pivot_fft_oid-pb.py.txt &
nohup python -u 012_pivot_highest178_oid-pb.py 0 > LOG/log_012_pivot_highest178_oid-pb.py.txt &
nohup python -u 013_pivot_top10_oid.py 0 > LOG/log_013_pivot_top10_oid.py.txt &
nohup python -u 014_pivot_detected_oid.py 0 > LOG/log_014_pivot_detected_oid.py.txt &
nohup python -u 015_pivot_detected_oid-pb.py 0 > LOG/log_015_pivot_detected_oid-pb.py.txt &
nohup python -u 017_pivot_highest60_oid-pb.py 0 > LOG/log_017_pivot_highest60_oid-pb.py.txt &


sleep 60

nohup python -u 018_pivot_highest60-120_oid-pb.py 0 > LOG/log_018_pivot_highest60-120_oid-pb.py.txt &
nohup python -u 019_month_change.py 0 > LOG/log_019_month_change.py.txt &
nohup python -u 020_pivot_highest60bf_oid-pb.py 0 > LOG/log_020_pivot_highest60bf_oid-pb.py.txt &
nohup python -u 021.py 0 > LOG/log_021.py.txt &
