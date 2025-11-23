@echo off
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 121314 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 383395482461 --fee 0.02915802764030608 --max-slippage 0.59593812580774 --fundamental-file-path data/ETH1.xlsx
python ttest.py --output_dir ttest_results
