

from transformers import TrOCRProcessor
import fastwer
from transformers import AutoTokenizer

tokenizer2 = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
processor.tokenizer = tokenizer2


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = fastwer.score_sent(pred_str, label_str, char_level=True)
    wer = fastwer.score_sent(pred_str, label_str)

    return {"cer": cer,"wer":wer}