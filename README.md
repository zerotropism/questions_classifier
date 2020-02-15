# Specification:
Script to classify Question and Sentence. The goal is to detect if asked question corresponds to mapped sentence. We need that for "Information Retriver". We used pytorch transformers from [There](https://github.com/huggingface/pytorch-transformers). The task "SST-2" corresponds for our needs, our new task called "QSC" for Question Sentence Classier" was dericated from this "SST-2" task.  

# Dataset:
The model is train and SQuaD v1.1 dataset. The original dataset is in JSON format, we will use the script from [There](https://gitlab.com/target-platform/datascience/utilities/tree/master/python/convert_SQUAD_for_BERT) to build our sampled_train.tsv, dev.tsv and test.tsv files necessary for traning, test and validations. As a classification task from BERT model we will use the following format "[CLS]\<question\>[SEP]\<sentence\>. You can a sample of this datasets in repo data, but real datasets need to be generated from utilities<br/>
The dataset will be composed to sentence to classify and corresponding label<br/>
Example(first question is false, second question is true)<br/>
sentence    label<br/>
[CLS] When was the Puerto Rican constitution written? [SEP] The phrase is usually used in local political debates, in polemic writing or in private conversations. It is rarely used by politicians themselves in a public context, although at certain times in Canadian history political parties have used other similarly loaded imagery. In the 1988 federal election, the Liberals asserted that the proposed Free Trade Agreement amounted to an American takeover of Canada—notably, the party ran an ad in which Progressive Conservative (PC) strategists, upon the adoption of the agreement, slowly erased the Canada-U.S. border from a desktop map of North America. Within days, however, the PCs responded with an ad which featured the border being drawn back on with a permanent marker, as an announcer intoned "Here's where we draw the line."    1<br/>
[CLS] What else can it produce? [SEP] An alloy of aluminium and gallium in pellet form added to water can be used to generate hydrogen. The process also produces alumina, but the expensive gallium, which prevents the formation of an oxide skin on the pellets, can be re-used. This has important potential implications for a hydrogen economy, as hydrogen can be produced on-site and does not need to be transported.    0<br/>


# Pre-requisites:
You need to install the folloing packages:<br/>
pip install pytorch-transformers<br/>
pip install sacremoses<br/>
pip install tensorboardX<br/>
pip install requirements.txt<br/>

# Training:
To train you need to run the following command line:<br/>
python ./qsc.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name QSC \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir data \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=64  \
    --per_gpu_train_batch_size=64  \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --output_dir ./model_qsc

python ./qsc.py --model_type bert --model_name_or_path bert-base-multilingual-uncased --task_name QSC --do_train --do_eval --do_lower_case --data_dir data/debug --max_seq_length 512 --per_gpu_eval_batch_size=64 --per_gpu_train_batch_size=64 --learning_rate 2e-5 --num_train_epochs 1.0 --output_dir ./model_qsc

# Prediction:
To predict you need to run following command line:<br/>
python ./qsc.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name QSC \
    --do_predict \
    --do_lower_case \
    --data_dir data/dev \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32  \
    --per_gpu_train_batch_size=32  \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --output_dir ./model_qsc