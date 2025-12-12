@echo off
python -u abides.py -c rmsc04 -t ETH -d 20251028 -s 12315 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 9500000000000 --fee 0.01 --max-slippage 0.05 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python ttest.py
