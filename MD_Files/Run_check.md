1. Is the process alive?

ps aux | grep train.py | grep -v grep
Output with a PID = running ✅
No output = it stopped (either finished or crashed)
2. Is it actually on the GPU this time? (important — verify it's not on CPU again)

nvidia-smi
Look for:

Your python3 process listed under Processes using ~15 GB
GPU-Util at ~90–99% once training starts (may show 0% for the first minute during data loading)
In the ps output above, %CPU should be modest (100–300%), not the 1800% you saw when it was stuck on CPU
3. Is it making progress?

tail -f ~/TMBL/run_emotion.log
Watch epochs appear live (you used -u, so it streams). Press Ctrl+C to stop watching — that does not stop the training.

Or check the per-epoch metrics log:


tail -n 15 ~/TMBL/checkpoints/log_emotion.txt
