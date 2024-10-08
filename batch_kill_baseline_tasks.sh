# ps aux | grep baseline_code | grep -v grep | awk '{print }' 
# ps aux | grep baseline_code | grep -v grep | awk '{print $2}' 
ps aux | grep baseline_code | grep -v grep | awk '{print $2}' | xargs kill -9