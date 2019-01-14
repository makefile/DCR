pids=$(ps au |grep [y]aml |awk '{print $2}')
if [ -z "$pids" ]; then
    echo 'no process'
else
    echo $(echo "$pids" | wc -l)
    echo $pids | xargs kill -9
fi
