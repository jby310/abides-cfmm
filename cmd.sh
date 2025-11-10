# run abides
# python -u abides.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_two_hour
python -u abides.py -c rmsc03 -t ETH -d 20251028 -s 1235 -l rmsc03_two_hour --end-time 10:00:00

python -u abides.py -c rmsc04 -t ETH -d 20251028 -s 1234 -l rmsc04_two_hour --end-time 10:00:00

# Plot using liquidity telemetry and explain what the plot does
cd util/plotting && python -u liquidity_telemetry.py ../../log/rmsc03_two_hour/EXCHANGE_AGENT.bz2 ../../log/rmsc03_two_hour/ORDERBOOK_ABM_FULL.bz2 \
-o rmsc03_two_hour.png -c configs/plot_09.30_11.30.json && cd ../../