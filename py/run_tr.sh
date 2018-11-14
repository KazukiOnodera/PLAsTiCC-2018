# nohup sh run_tr.sh &

python run.py 002_pivot_oid.py
python run.py 003_pivot_oid-pb.py
python run.py 004_pivot_oid-year.py

sleep 60

python run.py 005_pivot_oid-pb-year.py
python run.py 006_pivot_highest_oid.py
python run.py 007_pivot_lowest_oid.py
python run.py 008_pivot_diff_oid.py
python run.py 009_pivot_highest_oid-pb.py

sleep 60

python run.py 010_pivot_lowest_oid-year.py
python run.py 011_pivot_fft_oid-pb.py
python run.py 012_pivot_highest178_oid-pb.py
python run.py 013_pivot_top10_oid.py

sleep 60

python run.py 014_pivot_detected_oid.py
python run.py 015_pivot_detected_oid-pb.py
python run.py 017_pivot_highest60_oid-pb.py

sleep 60

python run.py 018_pivot_highest60-120_oid-pb.py
python run.py 019_month_change.py
python run.py 020_pivot_highest60bf_oid-pb.py
python run.py 021.py
