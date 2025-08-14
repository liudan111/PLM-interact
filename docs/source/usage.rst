Usage
==========================================

.. _usage:

Quick start
------------------------------------------

Predict a list PPIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To predict a list of PPIs, you can download pre-trained models from `Hugging Face <https://huggingface.co/danliu1226>`_.
Protein sequnce pair should be listed as follwing format:
Required Input:
   (--test_filepath): A CSV file with the following two columns:
    'query': The sequence of protein 1.
    'text': The sequence of protein 2.

   (--resume_from_checkpoint): the traiend model that can be downldoed from `Hugging Face <https://huggingface.co/danliu1226>`.

   (--offline_model_path): The 

   (--output_filepath): a path to save the results.

.. code-block:: bash
   torchrun --nproc_per_node=1 -m PLMinteract inference_PPI --seed 2 --batch_size_val 1 --test_filepath [a list of paired protein sequences] --resume_from_checkpoint [traiend model] --output_filepath $output_filepath --offline_model_path $offline_model_path --model_name esm2_t12_35M_UR50D --embedding_size 480 --max_length 1603 


There are 6 commands in PLM-interact package
----------------------------------------------
- `inference_PPI`: PPI prediction.
- `train_mlm`: Training PPI models using mask and binary classification losses.
- `train_binary`: Training PPI models using only binary classification loss.
- `predict_ddp`: Choose the best trained checkpoints by testing on the validation datasets and evaluate the model's performance on the test datasets.
- `mutation_train`:Fine-tuning in the binary mutation effect task.
- `mutation_predict`: Inference in the binary mutation effect task.


PPI prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
   usage: PLMinteract inference_PPI [-h] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                                    [--offline_model_path OFFLINE_MODEL_PATH] [--seed SEED]
                                    [--batch_size_val BATCH_SIZE_VAL] [--test_filepath TEST_FILEPATH]
                                    [--output_filepath OUTPUT_FILEPATH] [--model_name MODEL_NAME]
                                    [--embedding_size EMBEDDING_SIZE] [--max_length MAX_LENGTH]

   PPI prediction.

   options:
   -h, --help            show this help message and exit
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           If the training should continue from a checkpoint folder.
   --offline_model_path OFFLINE_MODEL_PATH
                           offline model path
   --seed SEED           seed
   --batch_size_val BATCH_SIZE_VAL
                           Input train batch size on each device (default: 32)
   --test_filepath TEST_FILEPATH
                           test_filepath
   --output_filepath OUTPUT_FILEPATH
                           output_filepath
   --model_name MODEL_NAME
                           model_name
   --embedding_size EMBEDDING_SIZE
                           embedding_size
   --max_length MAX_LENGTH
                           max_length


Training PPI models using mask and binary classification losses.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
   usage: PLMinteract train_mlm [-h] [--epochs EPOCHS] [--offline_model_path OFFLINE_MODEL_PATH]
                              [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--seed SEED] [--data DATA]
                              [--task_name TASK_NAME] [--batch_size_train BATCH_SIZE_TRAIN]
                              [--train_filepath TRAIN_FILEPATH] [--output_filepath OUTPUT_FILEPATH]
                              [--model_name MODEL_NAME] [--warmup_steps WARMUP_STEPS] [--embedding_size EMBEDDING_SIZE]
                              [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--max_length MAX_LENGTH]
                              [--weight_loss_mlm WEIGHT_LOSS_MLM] [--weight_loss_class WEIGHT_LOSS_CLASS]

   Training PPI models using mask and binary classification losses.

   options:
   -h, --help            show this help message and exit
   --epochs EPOCHS       Total epochs to train the model
   --offline_model_path OFFLINE_MODEL_PATH
                           offline ESM2 model path
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           If the training should continue from a checkpoint folder.
   --seed SEED           seed
   --data DATA           data name
   --task_name TASK_NAME
                           task_name
   --batch_size_train BATCH_SIZE_TRAIN
                           Input train batch size on each device (default: 16)
   --train_filepath TRAIN_FILEPATH
                           train filepath
   --output_filepath OUTPUT_FILEPATH
                           output filepath
   --model_name MODEL_NAME
                           ESM2 model name
   --warmup_steps WARMUP_STEPS
                           warmup steps
   --embedding_size EMBEDDING_SIZE
                           embedding size
   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                           gradient_accumulation_steps
   --max_length MAX_LENGTH
                           the max length of PPIs
   --weight_loss_mlm WEIGHT_LOSS_MLM
                           weight of mask loss
   --weight_loss_class WEIGHT_LOSS_CLASS
                           weight of classification loss


Training PPI models using only binary classification loss.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
   usage: PLMinteract train_binary [-h] [--epochs EPOCHS] [--offline_model_path OFFLINE_MODEL_PATH]
                                 [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--seed SEED] [--data DATA]
                                 [--task_name TASK_NAME] [--batch_size_train BATCH_SIZE_TRAIN]
                                 [--batch_size_val BATCH_SIZE_VAL] [--train_filepath TRAIN_FILEPATH]
                                 [--dev_filepath DEV_FILEPATH] [--test_filepath TEST_FILEPATH]
                                 [--output_filepath OUTPUT_FILEPATH] [--model_name MODEL_NAME]
                                 [--embedding_size EMBEDDING_SIZE] [--warmup_steps WARMUP_STEPS]
                                 [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--max_length MAX_LENGTH]
                                 [--evaluation_steps EVALUATION_STEPS] [--sub_samples SUB_SAMPLES]

   Training PPI models using only binary classification loss.

   options:
   -h, --help            show this help message and exit
   --epochs EPOCHS       Total epochs to train the model
   --offline_model_path OFFLINE_MODEL_PATH
                           offline model path
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           If the training should continue from a checkpoint folder.
   --seed SEED           seed
   --data DATA           data
   --task_name TASK_NAME
                           task_name
   --batch_size_train BATCH_SIZE_TRAIN
                           Input train batch size on each device (default: 16)
   --batch_size_val BATCH_SIZE_VAL
                           Input train batch size on each device (default: 32)
   --train_filepath TRAIN_FILEPATH
                           train_filepath
   --dev_filepath DEV_FILEPATH
                           dev_filepath
   --test_filepath TEST_FILEPATH
                           test_filepath
   --output_filepath OUTPUT_FILEPATH
                           output_filepath
   --model_name MODEL_NAME
                           model_name
   --embedding_size EMBEDDING_SIZE
                           embedding_size
   --warmup_steps WARMUP_STEPS
                           warmup_steps
   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                           gradient_accumulation_steps
   --max_length MAX_LENGTH
                           max_length
   --evaluation_steps EVALUATION_STEPS
                           evaluation_steps
   --sub_samples SUB_SAMPLES
                           sub_samples


Evaluation and test with multi nodes and multi GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

   usage: PLMinteract predict_ddp [-h] [--epochs EPOCHS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                                 [--offline_model_path OFFLINE_MODEL_PATH] [--seed SEED]
                                 [--batch_size_val BATCH_SIZE_VAL] [--dev_filepath DEV_FILEPATH]
                                 [--test_filepath TEST_FILEPATH] [--output_filepath OUTPUT_FILEPATH]
                                 [--model_name MODEL_NAME] [--embedding_size EMBEDDING_SIZE] [--max_length MAX_LENGTH]

   Choose the best trained checkpoints by testing on the validation datasets and evaluate the model's performance on the
   test datasets.

   options:
   -h, --help            show this help message and exit
   --epochs EPOCHS       Total epochs of trained model
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           If the training should continue from a checkpoint folder.
   --offline_model_path OFFLINE_MODEL_PATH
                           offline model path
   --seed SEED           seed
   --batch_size_val BATCH_SIZE_VAL
                           Input train batch size on each device (default: 32)
   --dev_filepath DEV_FILEPATH
                           dev_filepath
   --test_filepath TEST_FILEPATH
                           test_filepath
   --output_filepath OUTPUT_FILEPATH
                           output_filepath
   --model_name MODEL_NAME
                           model_name
   --embedding_size EMBEDDING_SIZE
                           embedding_size
   --max_length MAX_LENGTH
                           max_length


Fine-tuning in the binary mutation effect task.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
   usage: PLMinteract mutation_train [-h] [--epochs EPOCHS] [--offline_model_path OFFLINE_MODEL_PATH]
                                    [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--seed SEED]
                                    [--task_name TASK_NAME] [--batch_size_train BATCH_SIZE_TRAIN]
                                    [--batch_size_val BATCH_SIZE_VAL] [--train_filepath TRAIN_FILEPATH]
                                    [--dev_filepath DEV_FILEPATH] [--output_path OUTPUT_PATH] [--model_name MODEL_NAME]
                                    [--embedding_size EMBEDDING_SIZE] [--warmup_steps WARMUP_STEPS]
                                    [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--max_length MAX_LENGTH]
                                    [--weight_loss_mlm WEIGHT_LOSS_MLM] [--weight_loss_class WEIGHT_LOSS_CLASS]

   Fine-tuning in the binary mutation effect task.

   options:
   -h, --help            show this help message and exit
   --epochs EPOCHS       Total epochs to train the model
   --offline_model_path OFFLINE_MODEL_PATH
                           offline model path
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           If the training should continue from a checkpoint folder.
   --seed SEED           seed
   --task_name TASK_NAME
                           task_name
   --batch_size_train BATCH_SIZE_TRAIN
                           Input train batch size on each device (default: 16)
   --batch_size_val BATCH_SIZE_VAL
                           Input val batch size on each device (default: 16)
   --train_filepath TRAIN_FILEPATH
                           train_filepath
   --dev_filepath DEV_FILEPATH
                           dev_filepath
   --output_path OUTPUT_PATH
                           output_path
   --model_name MODEL_NAME
                           model_name
   --embedding_size EMBEDDING_SIZE
                           embedding_size
   --warmup_steps WARMUP_STEPS
                           warmup_steps
   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                           gradient_accumulation_steps
   --max_length MAX_LENGTH
                           max_length
   --weight_loss_mlm WEIGHT_LOSS_MLM
                           weight_loss_mlm
   --weight_loss_class WEIGHT_LOSS_CLASS
                           weight_loss_class


Inference in the binary mutation effect task.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash
   usage: PLMinteract mutation_predict [-h] [--offline_model_path OFFLINE_MODEL_PATH]
                                       [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--seed SEED]
                                       [--task_name TASK_NAME] [--batch_size_val BATCH_SIZE_VAL]
                                       [--test_filepath TEST_FILEPATH] [--output_path OUTPUT_PATH]
                                       [--model_name MODEL_NAME] [--embedding_size EMBEDDING_SIZE]
                                       [--max_length MAX_LENGTH] [--weight_loss_mlm WEIGHT_LOSS_MLM]
                                       [--weight_loss_class WEIGHT_LOSS_CLASS]

   Inference in the binary mutation effect task.

   options:
   -h, --help            show this help message and exit
   --offline_model_path OFFLINE_MODEL_PATH
                           offline model path
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           If the training should continue from a checkpoint folder.
   --seed SEED           seed
   --task_name TASK_NAME
                           task_name
   --batch_size_val BATCH_SIZE_VAL
                           Input train batch size on each device (default: 32)
   --test_filepath TEST_FILEPATH
                           test_filepath
   --output_path OUTPUT_PATH
                           output_path
   --model_name MODEL_NAME
                           model_name
   --embedding_size EMBEDDING_SIZE
                           embedding_size
   --max_length MAX_LENGTH
                           max_length
   --weight_loss_mlm WEIGHT_LOSS_MLM
                           weight_loss_mlm
   --weight_loss_class WEIGHT_LOSS_CLASS
                           weight_loss_class