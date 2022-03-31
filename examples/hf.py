from transformers import BartTokenizer, PegasusTokenizer
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration

IS_CNNDM = True # whether to use CNNDM dataset or XSum dataset
ARTICLE_TO_SUMMARIZE = "Manchester United superstar Cristiano Ronaldo scored his 806th career goal in Old Trafford,\
 breaking FIFA's all-time record for most goals in competitive matches in men's football history.\
 It was the second of three goals the Portuguese attacker scored during the game,\
 leading United to a 3-2 victory over Tottenham and finishing the day with 807 total career goals.\
 The previous FIFA goal record was held by Josef Bican, with 805 goals."

# Load our model checkpoints
if IS_CNNDM:
    model = BartForConditionalGeneration.from_pretrained('Yale-LILY/brio-cnndm-uncased')
    tokenizer = BartTokenizer.from_pretrained('Yale-LILY/brio-cnndm-uncased')
else:
    model = PegasusForConditionalGeneration.from_pretrained('Yale-LILY/brio-xsum-cased')
    tokenizer = PegasusTokenizer.from_pretrained('Yale-LILY/brio-xsum-cased')

max_length = 1024 if IS_CNNDM else 512
# generation example
if IS_CNNDM:
    article = ARTICLE_TO_SUMMARIZE.lower()
else:
    article = ARTICLE_TO_SUMMARIZE
inputs = tokenizer([article], max_length=max_length, return_tensors="pt", truncation=True)
# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
print(tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
