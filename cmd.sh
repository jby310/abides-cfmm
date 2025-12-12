# run abides
# python -u abides.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_two_hour
python -u abides.py -c rmsc03 -t ETH -d 20251110 -s 1234 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:35:00 --num-hybrid-agents 100 --fundamental-file-path data/ETH1.xlsx --r-bar 3611.0
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 1234 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:35:00 -k 50000000000 --fee 0.001 --max-slippage 0.05 --num-hybrid-agents 100 --fundamental-file-path data/ETH1.xlsx --r-bar 3611.0
python ttest.py
python plot.py
python threshold3D.py

python -u abides.py -c rmsc03 -t ETH -d 20251028 -s 5678 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:40:00 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python -u abides.py -c rmsc04 -t ETH -d 20251028 -s 5678 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 1e12 --fee 0.001 --max-slippage 0.05 --num-hybrid-agents 100 --fundamental-file-path data/BIT.xlsx --r-bar 113994.6305
python ttest.py
python plot.py

python -u abides.py -c rmsc03 -t ETH -d 20251110 -s 1234 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:35:00 --num-hybrid-agents 100 --fundamental-file-path data/BNB.xlsx --r-bar 550.0
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 12315 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:35:00 -k 9000000000 --fee 0.001 --max-slippage 0.05 --num-hybrid-agents 100 --fundamental-file-path data/BNB.xlsx --r-bar 550.0
python ttest.py
python plot.py
python threshold3D.py

python -u abides.py -c rmsc03 -t ETH -d 20251110 -s 1234 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:40:00 --num-hybrid-agents 100 --fundamental-file-path data/DOGE.xlsx --r-bar 0.2
python -u abides.py -c rmsc04 -t ETH -d 20251110 -s 12315 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:40:00 -k 900000 --fee 0.001 --max-slippage 0.01 --num-hybrid-agents 100 --fundamental-file-path data/DOGE.xlsx --r-bar 0.2
python ttest.py
python plot.py
python threshold3D.py


# Plot using liquidity telemetry and explain what the plot does
cd util/plotting && python -u liquidity_telemetry.py ../../log/rmsc03_two_hour/EXCHANGE_AGENT.bz2 ../../log/rmsc03_two_hour/ORDERBOOK_ABM_FULL.bz2 \
-o rmsc03_two_hour.png -c configs/plot_09.30_11.30.json && cd ../../