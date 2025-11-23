@echo off
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 5678 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 13257314918 --fee 0.0013481316372955159 --max-slippage 0.08899477733192865 --fundamental-file-path data/ETH1.xlsx
python ttest.py --output_dir ttest_results
