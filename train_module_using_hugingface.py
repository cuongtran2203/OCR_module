from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor
import pandas as pd
from dataloader import OCR_VN
from transformers import VisionEncoderDecoderModel
from metrics import *
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator
import os
os.makedirs("results",exist_ok=True)
if __name__ == "__main__":
    csv_path = ''
    EPOCHS = 1000
    BATCH_SIZE = 8
    df = pd.read_csv(csv_path)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    # Define Datasets
    train_df, test_df = train_test_split(df, test_size=0.2)
    # we reset the indices to start from zero
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    train_dataset = OCR_VN(root_dir='data_ocr',
                           df=train_df,
                           processor=processor)
    eval_dataset = OCR_VN(root_dir='data_ocr',
                           df=test_df,
                           processor=processor)
    
    #Define model
        
  
    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 512
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    # Trainer
    training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    num_train_epochs= EPOCHS,
    evaluation_strategy="steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    fp16=True, 
    output_dir="results",
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
    )
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()