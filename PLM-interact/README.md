# Mutation effect prediction task

## Training: 
### Required Input (--train_filepath): A CSV file with the following four columns:
    query: The sequence of protein 1.
    text: The sequence of protein 2.

## Prediction: 
### Required Input (--test_filepath): A CSV file with the following four columns:
    query: The sequence of protein 1.
    text: The sequence of protein 2.
    label: The ground truth label, where 1 indicates a positive interaction and 0 indicates a negative one.
