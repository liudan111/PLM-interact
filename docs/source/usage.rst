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
   (--test_filepath): A CSV file with the following two columns: 'query': The sequence of protein 1, and 'text': The sequence of protein 2.

   (--resume_from_checkpoint): the traiend model that can be downldoed from `Hugging Face <https://huggingface.co/danliu1226>`.

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
------------
.. code-block:: bash
   usage: PLMinteract inference_PPI [-h] [--seed SEED] --test_filepath TEST_FILEPATH --output_filepath OUTPUT_FILEPATH
                                 --resume_from_checkpoint RESUME_FROM_CHECKPOINT [--batch_size_val BATCH_SIZE_VAL]
                                 [--max_length MAX_LENGTH] --offline_model_path OFFLINE_MODEL_PATH --model_name MODEL_NAME
                                 --embedding_size EMBEDDING_SIZE

   PPI prediction.

   options:
   -h, --help            show this help message and exit
   --seed SEED           Random seed for reproducibility (default: 2).
   --test_filepath TEST_FILEPATH
                           Path to the test dataset (CSV format).
   --output_filepath OUTPUT_FILEPATH
                           Path to save the prediction results.
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           Path to a trained model (default: None).
   --batch_size_val BATCH_SIZE_VAL
                           The validation batch size on each device (default: 16).
   --max_length MAX_LENGTH
                           Maximum sequence length for tokenizing paired proteins (default: 1603).
   --offline_model_path OFFLINE_MODEL_PATH
                           Path to a locally stored ESM-2 model.
   --model_name MODEL_NAME
                           Choose the ESM-2 model to load (esm2_t12_35M_UR50D / esm2_t33_650M_UR50D).
   --embedding_size EMBEDDING_SIZE
                           Set embedding vector size based on the selected ESM-2 model (480 / 1280).


Training PPI models using mask and binary classification losses
------------
.. code-block:: bash

   usage: PLMinteract train_mlm [-h] [--seed SEED] [--data DATA] [--task_name TASK_NAME] --train_filepath TRAIN_FILEPATH
                              --output_filepath OUTPUT_FILEPATH [--epochs EPOCHS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                              [--warmup_steps WARMUP_STEPS] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                              [--weight_loss_mlm WEIGHT_LOSS_MLM] [--weight_loss_class WEIGHT_LOSS_CLASS] [--max_length MAX_LENGTH]
                              [--batch_size_train BATCH_SIZE_TRAIN] --offline_model_path OFFLINE_MODEL_PATH --model_name MODEL_NAME
                              --embedding_size EMBEDDING_SIZE

   Training PPI models using mask and binary classification losses.

   options:
   -h, --help            show this help message and exit
   --seed SEED           Random seed for reproducibility (default: 2).
   --data DATA           Set the dataset name (e.g., cross_species)(default: "").
   --task_name TASK_NAME
                           Set the task name (e.g., 1vs10, 1vs1)(default: "").

   Input data and path of output results:
   --train_filepath TRAIN_FILEPATH
                           Path to the training dataset (CSV format).
   --output_filepath OUTPUT_FILEPATH
                           Path to save trained model checkpoints and training results.

   PLM-interact setting:
   --epochs EPOCHS       Total number of training epochs (default: 10)
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           Path to a checkpoint to resume training from, if continuing a previous run.
   --warmup_steps WARMUP_STEPS
                           Number of warmup steps for the learning rate scheduler (default: 2000).
   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                           Number of steps to accumulate gradients before performing an optimizer step (default: 8).
   --weight_loss_mlm WEIGHT_LOSS_MLM
                           Weight applied to the masked language modeling (MLM) loss (default: 1).
   --weight_loss_class WEIGHT_LOSS_CLASS
                           Weight applied to the classification loss (default: 10).
   --max_length MAX_LENGTH
                           Maximum sequence length for tokenizing paired proteins (default: 1603).
   --batch_size_train BATCH_SIZE_TRAIN
                           The training batch size on each device (default: 16).

   ESM2 model loading:
   --offline_model_path OFFLINE_MODEL_PATH
                           Path to a locally stored ESM-2 model.
   --model_name MODEL_NAME
                           Choose the ESM-2 model to load (esm2_t12_35M_UR50D / esm2_t33_650M_UR50D).
   --embedding_size EMBEDDING_SIZE
                           Set embedding vector size based on the selected ESM-2 model (480 / 1280).


Training PPI models using only binary classification loss.
------------
.. code-block:: bash
      usage: PLMinteract train_binary [-h] [--seed SEED] [--data DATA] [--task_name TASK_NAME] --train_filepath TRAIN_FILEPATH
                                    --dev_filepath DEV_FILEPATH --test_filepath TEST_FILEPATH --output_filepath OUTPUT_FILEPATH
                                    [--epochs EPOCHS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--warmup_steps WARMUP_STEPS]
                                    [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--evaluation_steps EVALUATION_STEPS]
                                    [--sub_samples SUB_SAMPLES] [--max_length MAX_LENGTH] [--batch_size_train BATCH_SIZE_TRAIN]
                                    [--batch_size_val BATCH_SIZE_VAL] --offline_model_path OFFLINE_MODEL_PATH --model_name MODEL_NAME
                                    --embedding_size EMBEDDING_SIZE

      Fine-tuning in the binary mutation effect task

      options:
      -h, --help            show this help message and exit
      --seed SEED           Random seed for reproducibility (default: 2).
      --data DATA           Set the dataset name (e.g., cross_species)(default: "").
      --task_name TASK_NAME
                              Set the task name (e.g., binary)(default: "").

      Input data and path of output results:
      --train_filepath TRAIN_FILEPATH
                              Path to the training dataset (CSV format).
      --dev_filepath DEV_FILEPATH
                              Path to the validation dataset (CSV format).
      --test_filepath TEST_FILEPATH
                              Path to the test dataset (CSV format).
      --output_filepath OUTPUT_FILEPATH
                              Path to save trained model checkpoints and training results.

      PLM-interact setting:
      --epochs EPOCHS       Total number of training epochs (default: 10).
      --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                              Path to a checkpoint to resume training from, if continuing a previous run.
      --warmup_steps WARMUP_STEPS
                              Number of warmup steps for the learning rate scheduler (default: 2000).
      --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                              Number of steps to accumulate gradients before performing an optimizer step (default: 8).
      --evaluation_steps EVALUATION_STEPS
                              Perform evaluation every N steps during training (default: 5000).
      --sub_samples SUB_SAMPLES
                              Number of subsamples to use for evaluation (default: 128).
      --max_length MAX_LENGTH
                              Maximum sequence length for tokenizing paired proteins (default: 1603).
      --batch_size_train BATCH_SIZE_TRAIN
                              The training batch size on each device (default: 16).
      --batch_size_val BATCH_SIZE_VAL
                              The validation batch size on each device (default: 16).

      ESM2 model loading:
      --offline_model_path OFFLINE_MODEL_PATH
                              Path to a locally stored ESM-2 model.
      --model_name MODEL_NAME
                              Choose the ESM-2 model to load (esm2_t12_35M_UR50D / esm2_t33_650M_UR50D).
      --embedding_size EMBEDDING_SIZE
                              Set embedding vector size based on the selected ESM-2 model (480 / 1280).

Evaluation and test with multi nodes and multi GPUs
------------
.. code-block:: bash
   usage: PLMinteract predict_ddp [-h] [--seed SEED] --dev_filepath DEV_FILEPATH --test_filepath TEST_FILEPATH --output_filepath
                                 OUTPUT_FILEPATH [--epochs EPOCHS] [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                                 [--batch_size_val BATCH_SIZE_VAL] [--max_length MAX_LENGTH] --offline_model_path OFFLINE_MODEL_PATH
                                 --model_name MODEL_NAME --embedding_size EMBEDDING_SIZE

   Choose the best trained checkpoints by testing on the validation datasets and evaluate the model's performance on the test
   datasets.

   options:
   -h, --help            show this help message and exit
   --seed SEED           Random seed for reproducibility (default: 2).

   Input data and output results:
   --dev_filepath DEV_FILEPATH
                           Path to the validation dataset (CSV format).
   --test_filepath TEST_FILEPATH
                           Path to the test dataset (CSV format).
   --output_filepath OUTPUT_FILEPATH
                           Path to save validation and test results.

   PLM-interact setting:
   --epochs EPOCHS       Total epochs of trained models (default: 10).
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           Path to trained models(default: None).
   --batch_size_val BATCH_SIZE_VAL
                           The validation batch size on each device (default: 16)
   --max_length MAX_LENGTH
                           Maximum sequence length for tokenizing paired proteins (default: 1603).

   ESM2 model loading:
   --offline_model_path OFFLINE_MODEL_PATH
                           Path to a locally stored ESM-2 model.
   --model_name MODEL_NAME
                           Choose the ESM-2 model to load (esm2_t12_35M_UR50D / esm2_t33_650M_UR50D).
   --embedding_size EMBEDDING_SIZE
                           Set embedding vector size based on the selected ESM-2 model (480 / 1280).


Fine-tuning in the binary mutation effect task.
------------
.. code-block:: bash
   usage: PLMinteract mutation_train [-h] [--seed SEED] [--task_name TASK_NAME] --train_filepath TRAIN_FILEPATH --dev_filepath
                                  DEV_FILEPATH --output_path OUTPUT_PATH [--epochs EPOCHS]
                                  [--resume_from_checkpoint RESUME_FROM_CHECKPOINT] [--warmup_steps WARMUP_STEPS]
                                  [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS] [--weight_loss_mlm WEIGHT_LOSS_MLM]
                                  [--weight_loss_class WEIGHT_LOSS_CLASS] [--max_length MAX_LENGTH]
                                  [--batch_size_train BATCH_SIZE_TRAIN] [--batch_size_val BATCH_SIZE_VAL] --offline_model_path
                                  OFFLINE_MODEL_PATH --model_name MODEL_NAME --embedding_size EMBEDDING_SIZE

   Predict mutant effects in human PPIs.

   options:
   -h, --help            show this help message and exit
   --seed SEED           Random seed for reproducibility. (default: 2)
   --task_name TASK_NAME
                           Set the task name (e.g., mutation_effects_training)(default: "")

   Input data and path of output results:
   --train_filepath TRAIN_FILEPATH
                           Path to the training dataset (CSV format)
   --dev_filepath DEV_FILEPATH
                           Path to the validation dataset (CSV format)
   --output_path OUTPUT_PATH
                           Path to save trained model checkpoints and training results

   PLM-interact setting:
   --epochs EPOCHS       Total number of training epochs (default: 50).
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           Path to a checkpoint to resume training from, if continuing a previous run
   --warmup_steps WARMUP_STEPS
                           Number of warmup steps for the learning rate scheduler (default: 2000)
   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                           Number of steps to accumulate gradients before performing an optimizer step (default: 8)
   --weight_loss_mlm WEIGHT_LOSS_MLM
                           Weight applied to the masked language modeling (MLM) loss (default: 1)
   --weight_loss_class WEIGHT_LOSS_CLASS
                           Weight applied to the classification loss (default: 10)
   --max_length MAX_LENGTH
                           Maximum sequence length for tokenizing paired proteins (default: 1603)
   --batch_size_train BATCH_SIZE_TRAIN
                           The training batch size on each device (default: 16)
   --batch_size_val BATCH_SIZE_VAL
                           The validation batch size on each device (default: 16)

   ESM-2 model loading:
   --offline_model_path OFFLINE_MODEL_PATH
                           Path to a locally stored ESM-2 model
   --model_name MODEL_NAME
                           Choose the ESM-2 model to load (esm2_t12_35M_UR50D / esm2_t33_650M_UR50D)
   --embedding_size EMBEDDING_SIZE
                           Set embedding vector size based on the selected ESM-2 model (480 / 1280)


Inference in the binary mutation effect task.
------------

.. code-block:: bash

   usage: PLMinteract mutation_predict [-h] [--seed SEED] [--task_name TASK_NAME] --test_filepath TEST_FILEPATH --output_path
                                    OUTPUT_PATH --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                                    [--weight_loss_mlm WEIGHT_LOSS_MLM] [--weight_loss_class WEIGHT_LOSS_CLASS]
                                    [--max_length MAX_LENGTH] [--batch_size_val BATCH_SIZE_VAL] --offline_model_path
                                    OFFLINE_MODEL_PATH --model_name MODEL_NAME --embedding_size EMBEDDING_SIZE

   Inference in the binary mutation effect task

   options:
   -h, --help            show this help message and exit
   --seed SEED           Random seed for reproducibility (default: 2).
   --task_name TASK_NAME
                           Set the task name (e.g., mutation_effects_pre)(default: "").

   Input data and path of output results:
   --test_filepath TEST_FILEPATH
                           Path to the input CSV file for testing.
   --output_path OUTPUT_PATH
                           Path to save prediction results.

   PLM-interact parameters:
   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                           Path to a trained model.
   --weight_loss_mlm WEIGHT_LOSS_MLM
                           Weight applied to the masked language modeling (MLM) loss (default: 1).
   --weight_loss_class WEIGHT_LOSS_CLASS
                           Weight applied to the classification loss (default: 10).
   --max_length MAX_LENGTH
                           Maximum sequence length for tokenizing paired proteins (default: 1603).
   --batch_size_val BATCH_SIZE_VAL
                           The validation batch size on each device (default: 16).

   ESM2 model loading:
   --offline_model_path OFFLINE_MODEL_PATH
                           Path to a locally stored ESM-2 model.
   --model_name MODEL_NAME
                           Choose the ESM-2 model to load (esm2_t12_35M_UR50D / esm2_t33_650M_UR50D).
   --embedding_size EMBEDDING_SIZE
                           Set embedding vector size based on the selected ESM-2 model (480 / 1280).