### Ae-trainer:
Trains auto-encoders by fine-tuning pretrained encoder and decoder models to create latent representations.
#### Usage:
First run
`python prep_data.py` with the appropriate args to generate the data.
Then update `train_config.yaml` according to your needs.
Finally run `python ae_trainer.py --config train_config.yaml`
#### Results:
Currently acheives around ~3 validation loss (22 perplexity) with the config provided, but reconstructions are still poor.
#### Models:
Code to support the following models:
- Modern BERT
- GPT-2 (needs to be updated)
- llama3
- qwen2.5
Additionaly supports cross attention module

### sonar_eval:
Evaluates the encoding performance of the sonar model on various tasks.
