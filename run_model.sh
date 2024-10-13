type=$1
if [ $type == "train" ]; then
    python train.py $2 $3
elif [ $type == "test" ]; then
    python test.py $2 $3 $4
else
    echo "Invalid type. must be train or test"
fi
