model_path="./models"
output_path="./outputs"
log_path="./logs"
model_file="./models/BaSNet_model_best.pkl"
video_path="./video/"
feature_path="./data/"

conda activate BASNET

python -W ignore ./Neural_Network.py --model_path ${model_path} --output_path ${output_path} --log_path ${log_path} --model_file ${model_file} --video_path ${video_path} --feature_path ${feature_path}
