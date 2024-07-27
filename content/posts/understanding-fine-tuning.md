---
title: "Understanding Fine Tuning"
date: 2024-07-27T12:00:00+00:00
draft: false
---

# Experiments in Fine Tuning

The high-level idea of fine tuning is fairly straightforward: re-training a model based on some set of new data should be able to give the model new knowledge or skills. Many services offer “out-of-the-box” fine-tuning on both open and closed models, and they make the process extremely simple: you submit a dataset, they train the model and host it. However, how this works under the hood was a mystery for me.

## Structure of GPT2

Roughly speaking, the main brain of GPT2 is made up of 12 neural networks (transformer blocks) stacked together, where each module contains both an attention mechanism and an MLP, along with other components. Other than that, there are a few other layers at the input and output of the transformer blocks. 

```python
Input
  |
[Embedding Layer]
  |
[Positional Encoding]
  |
[Transformer Block 1]  <-- Contains Attention + MLP
  |
[Transformer Block 2]  <-- Contains Attention + MLP
  |
  ...
  |
[Transformer Block 12] <-- Contains Attention + MLP
  |
[Final Layer Norm]
  |
[Output Layer]
```

Each of these layers and transformer blocks have weights, which we can visualize as tensors. 

```python
Initial Embed: 
 tensor([[-0.1101, -0.0393,  0.0331,  ..., -0.1364,  0.0151,  0.0453],
        [ 0.0403, -0.0486,  0.0462,  ...,  0.0861,  0.0025,  0.0432],
        [-0.1275,  0.0479,  0.1841,  ...,  0.0899, -0.1297, -0.0879],
        ...,
        [-0.0445, -0.0548,  0.0123,  ...,  0.1044,  0.0978, -0.0695],
        [ 0.1860,  0.0167,  0.0461,  ..., -0.0963,  0.0785, -0.0225],
        [ 0.0514, -0.0277,  0.0499,  ...,  0.0070,  0.1552,  0.1207]])
Attn 0: 
 tensor([[-0.4738, -0.2614, -0.0978,  ...,  0.0513, -0.0584,  0.0250],
        [ 0.0874,  0.1473,  0.2387,  ..., -0.0525, -0.0113, -0.0156],
        [ 0.0039,  0.0695,  0.3668,  ...,  0.1143,  0.0363, -0.0318],
        ...,
        [-0.2592, -0.0164,  0.1991,  ...,  0.0095, -0.0516,  0.0319],
        [ 0.1517,  0.2170,  0.1043,  ...,  0.0293, -0.0429, -0.0475],
        [-0.4100, -0.1924, -0.2400,  ..., -0.0046,  0.0070,  0.0198]])
Attn 1: 
 tensor([[-0.2906,  0.3057,  0.0302,  ..., -0.0057, -0.0582, -0.0061],
        [-0.3272,  0.2420,  0.2140,  ..., -0.0100,  0.1192, -0.1672],
        [-0.2679,  0.1188, -0.2670,  ...,  0.1511,  0.0671,  0.0421],
        ...,
        [-0.0284,  0.4304, -0.1394,  ...,  0.0283,  0.1013, -0.0133],
        [ 0.1730,  0.0967,  0.0262,  ..., -0.0811,  0.0632, -0.0570],
        [ 0.0422,  0.1598, -0.2512,  ..., -0.0145, -0.0245,  0.0788]])
.... 

Attn 12: 
 tensor([[-0.2659,  0.0279,  0.0728,  ..., -0.1061,  0.0058,  0.1481],
        [ 0.0896, -0.2727,  0.1485,  ..., -0.0797, -0.0038,  0.0260],
        [ 0.0594,  0.1710, -0.3967,  ..., -0.0204,  0.1682, -0.0434],
        ...,
        [ 0.2275,  0.2548, -0.0267,  ..., -0.0866, -0.0891, -0.0625],
        [ 0.1615,  0.0674,  0.1885,  ..., -0.1322, -0.1471, -0.1285],
        [-0.3938, -0.0045, -0.0314,  ..., -0.1211, -0.0433,  0.2487]])
```

Since GPT2 was primarily trained on English data, I wanted to see if we could fine-tune multi-lingual capability into the vanilla GPT2 model. Testing some Chinese text-completion prompts, GPT2 produced the following answers

```python
Prompt: 今天的天气
Generated: 今天的天气想下。

非常打那是这样的现在住时间传有是你个一讲的恐性,我们到他什么多解告了。 "This is a new country. It is the first time in history that the United

Prompt: 人工智能的未来
Generated: 人工智能的未来.

"No, I'm sorry."
—
.

The city was a bit more peaceful than before, as well as the two countries. However, there was still the issue of the undead, and that had to be solved. In order to prevent that, the city's main residence was the palace, which was located in the middle of a courtyard. At this moment, a young man sat at the
```

It seems like GPT2 was trained on *some* Chinese text data, as it is capable of producing Chinese characters as output. But it seems to be complete slop, and the model adds random English sentences in the response as well. Would GPT2 fine-tuned on Chinese text be capable of producing better outputs?

## Fine Tuning

I downloaded a [small dataset of Chinese text](https://huggingface.co/datasets/Delius/ChineseWebNovel) from HuggingFace and attempted to fine-tune the model on this dataset. Stripping out a bunch of informational text like “publication time” and “next page” in the dataset, I got a Chinese text dataset of ~72m characters which was approximately 200MB. 

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Add padding token to tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize dataset
train_dataset = load_and_subsample_dataset("chinese_responses.txt", tokenizer)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-chinese-fine-tuned",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir='./logs',
    logging_steps=10,  # Reduced due to smaller dataset
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

#Train the model
trainer.train()

# Save the model
trainer.save_model()
tokenizer.save_pretrained("./gpt2-chinese-fine-tuned")
```

The training took too long (~40 hours) on my Macbook Pro so I decided to just subsample the dataset to make it quicker. This only used 1/1000 of the original dataset, and used random characters. This likely made the training worse because I’m using “random” subsamples of the data, meaning the sequences of words will not be coherent at all. The subsampling is also done using a tokenizer function, which I simply used the default English tokenizer. 

```python
from datasets import load_dataset
import random

def load_and_subsample_dataset(file_path, tokenizer, fraction=0.001):
    # Load the full dataset
    dataset = load_dataset('text', data_files={'train': file_path})['train']
    
    # Calculate the number of samples to keep
    num_samples = max(1, int(len(dataset) * fraction))
    
    # Randomly sample the dataset
    sampled_indices = random.sample(range(len(dataset)), num_samples)
    sampled_dataset = dataset.select(sampled_indices)
    
    # Tokenize the sampled dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)
    
    tokenized_dataset = sampled_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    return tokenized_dataset
```

Since I used a small subsample of the data, the training run only took 5 minutes. The loss was bad and plateaued around 2.5, likely because the data was pretty bad due to the subsampling and the use of the English tokenizer. 

```python
Step	Training Loss
10	3.525200
20	3.177700
30	3.118300
40	3.104500
50	3.040100
...
1530	2.522100
1540	2.405500
1550	2.506000
```

## Testing the Fine-Tuned Model

Despite the poor loss and short training run, surprisingly, the model seemed to be more fluent than before in Chinese!

```python
Prompt: 今天的天气
Generated: 今天的天气，但是一个拿体ｚ“你面的话ﾌ不过ￌ说道。”些我没有做顾下ﲌ其中是主觉何们把怒知違。那两�

Prompt: 中国的经济
Generated: 中国的经济起了西间，却年不由交出计谈ﻌ那个地方沉等急发把儿ｌ跟以这可他的做于探笑道ﾌ每

Prompt: 人工智能的未来
Generated: 人工智能的未来少无人，果然只是我说这个人都是自己是在清楚的女人女消ﻌ而且所那么头殿处的让或者ｌ一下聊�
```

The model now generates 100% Chinese characters instead of the previous mix of English and Chinese. However, the output sentences still made no sense gramatically – it was just a bunch of random Chinese words. By improving the dataset and doing a longer training run, we could get probably get more coherent outputs from the fine-tuned model.

I also tried to observe the differences in the weights from before and after fine-tuning. 

```python
import torch
import numpy as np

def analyze_embeddings(embed1, embed2):
    diff = embed1 - embed2
    
    print(f"Mean absolute difference: {torch.abs(diff).mean().item():.6f}")
    print(f"Max absolute difference: {torch.abs(diff).max().item():.6f}")
    print(f"Standard deviation of differences: {diff.std().item():.6f}")
    
    cosine_sim = torch.nn.functional.cosine_similarity(embed1, embed2)
    print(f"Mean cosine similarity: {cosine_sim.mean().item():.6f}")
    
    euclidean_dist = torch.norm(embed1 - embed2, dim=1)
    print(f"Mean Euclidean distance: {euclidean_dist.mean().item():.6f}")

# Assuming embed1 and embed2 are your tensor variables
analyze_embeddings(attn_1, attn_1_cn)

Mean absolute difference: 0.001576
Max absolute difference: 0.027599
Standard deviation of differences: 0.002691
Mean cosine similarity: 0.999807
Mean Euclidean distance: 0.063901
```

I actually expected the entire set of weights to change with the new data, but they were actually almost identical to the weights of the original GPT2 model. From the Attention 1 layer, the mean cosine similarity was 0.9998, and the mean Euclidean distance was 0.06, suggesting that the vectors were extremely similar.

This suggests that the model did not really “learn” much from the fine-tuning – it just adjusted its weights ever so slightly. Models like GPT2 and more advanced ones already “learn” a bunch of things about language, sentence structure, grammar, and so on, so fine-tuning it on new data that gives more examples of what it already knows would not cause it to update its weights significantly. 

I might redo this training run on the cloud with the full dataset and observe the results. I would also need to create a way to evaluate how well the model does based on the new dataset vs current dataset – but that is another can of worms for another day.