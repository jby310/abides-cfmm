@echo off
python -u abides.py -c rmsc04 -t ETH -d 20251028 -s 1235 -l rmsc04_two_hour --end-time 09:40:00 -k 100000000 --fee 0.008
python ttest.py --output_dir ttest_results
