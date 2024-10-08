# ps aux | grep baseline_code | grep -v grep | awk '{print }' 
# ps aux | grep baseline_code | grep -v grep | awk '{print $2}' 
ps aux | grep baseline_code | grep -v grep | awk '{print $2}' | xargs kill -9
ps aux | grep core_model | grep -v grep | awk '{print $2}' | xargs kill -9
