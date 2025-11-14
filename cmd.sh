# run abides
# python -u abides.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_two_hour
python -u abides.py -c rmsc03 -t ETH -d 20251114 -s 1235 -l rmsc03_two_hour --start-time 09:30:00 --end-time 09:45:00 --fundamental-file-path data/ETH.xlsx
python -u abides.py -c rmsc04 -t ETH -d 20251114 -s 1235 -l rmsc04_two_hour --start-time 09:30:00 --end-time 09:45:00 -k 10000000 --fee 0.008 --fundamental-file-path data/ETH.xlsx
python ttest.py

# Plot using liquidity telemetry and explain what the plot does
cd util/plotting && python -u liquidity_telemetry.py ../../log/rmsc03_two_hour/EXCHANGE_AGENT.bz2 ../../log/rmsc03_two_hour/ORDERBOOK_ABM_FULL.bz2 \
-o rmsc03_two_hour.png -c configs/plot_09.30_11.30.json && cd ../../