@echo off
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 1235 -l rmsc04_two_hour --start-time 09:30:00 --end-time 16:00:00 -k 100000 --fee 0.001 --max-slippage 0.1 --fundamental-file-path data/ETH1.xlsx
python ttest.py --output_dir ttest_results
