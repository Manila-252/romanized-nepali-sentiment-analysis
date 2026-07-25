# Evaluating DistilBERT for Sentiment Classification of Romanized Nepali Restaurant Reviews

**An experimental study investigating transfer learning for sentiment analysis in a low-resource language setting using a transformer-based language model.**

---

## Project Summary

Transformer-based language models have achieved remarkable success across many Natural Language Processing (NLP) tasks. However, their application to low-resource languages remains challenging due to limited annotated datasets, inconsistent writing conventions, and computational constraints.

This project investigates the effectiveness of **DistilBERT**, an English-pretrained transformer model, for three-class sentiment classification of **Romanized Nepali restaurant reviews**. The objective was to evaluate whether transfer learning could provide reliable sentiment prediction despite limited labeled data and informal transliterated text.

During development, the project initially explored parameter-efficient fine-tuning approaches for **Llama 2**, including **LoRA** and **QLoRA**. Due to GPU memory limitations within the available computing environment, the experimental design was revised to use DistilBERT—a computationally efficient transformer architecture well suited for supervised sequence classification.

The final model achieved approximately **82.8% test accuracy**, demonstrating that compact transformer models can effectively address sentiment analysis tasks in a low-resource NLP setting.

---

# Key Highlights

- Fine-tuned **DistilBERT** for three-class sentiment classification.
- Evaluated sentiment prediction on **607 manually labeled Romanized Nepali restaurant reviews**.
- Explored parameter-efficient fine-tuning concepts (LoRA and QLoRA) before selecting a more practical architecture.
- Achieved approximately **82.8% classification accuracy** on unseen test data.
- Investigated transfer learning for a low-resource language with informal transliterated text.

---

# Research Motivation

Natural language understanding for low-resource languages remains an important challenge in modern NLP research. While transformer architectures have demonstrated exceptional performance for English and other high-resource languages, significantly fewer resources exist for Romanized Nepali.

Romanized Nepali introduces additional complexity because speakers write Nepali using the Latin alphabet without standardized spelling conventions. User-generated text frequently contains inconsistent transliterations, abbreviations, mixed English vocabulary, and informal writing styles.

This project explores whether transfer learning can overcome these challenges using a lightweight transformer architecture.

---

# Research Question

**How effectively can an English-pretrained DistilBERT model perform three-class sentiment classification on Romanized Nepali restaurant reviews?**

---

# Dataset

**Source**

- Hugging Face: `amirpoudel/nepal-romanized-restaurant-reviews`

**Task**

- Positive sentiment
- Neutral sentiment
- Negative sentiment

**Dataset Characteristics**

- 607 manually labeled reviews
- Romanized Nepali text
- Mixed Nepali-English vocabulary
- User-generated restaurant reviews
- Low-resource language setting

| Split | Samples |
|-------|---------:|
| Training | 485 |
| Testing | 122 |

---

# Methodology

The experimental workflow consisted of:

1. Data preprocessing and label encoding
2. Train-test split
3. Tokenization using DistilBERT tokenizer
4. Fine-tuning DistilBERT for sequence classification
5. Model evaluation using classification metrics
6. Qualitative analysis of prediction performance

The implementation was developed using the Hugging Face Transformers ecosystem with PyTorch as the underlying deep learning framework.

---

# Experimental Design Decisions

One of the primary objectives of this project was to understand how model selection should balance predictive performance with computational feasibility.

The project initially investigated fine-tuning **Llama 2** using **LoRA** and **QLoRA**. While these approaches were explored conceptually and experimentally, GPU memory limitations prevented stable fine-tuning within the available hardware environment.

Rather than increasing computational complexity, the study adopted **DistilBERT**, which provided:

- Lower computational requirements
- Faster training
- Efficient inference
- Strong performance for sequence classification
- Practical deployment on limited hardware

This experience reinforced the importance of selecting models that align with both research objectives and available computational resources.

---

# Experimental Configuration

| Parameter | Value |
|-----------|--------|
| Base Model | DistilBERT Base Uncased |
| Epochs | 6 |
| Learning Rate | 2e-5 |
| Batch Size | 4 |
| Maximum Sequence Length | 128 |
| Weight Decay | 0.01 |
| Mixed Precision | FP16 |

---

# Results

The fine-tuned model achieved the following performance on the held-out test set.

| Metric | Score |
|---------|-------:|
| Accuracy | **82.79%** |
| Precision | **0.83** |
| Recall | **0.83** |
| F1 Score | **0.83** |

The model demonstrated balanced performance across positive, neutral, and negative sentiment classes while successfully generalizing to previously unseen Romanized Nepali reviews.

---

# Key Findings

- Transfer learning proved effective despite the relatively small dataset.
- Compact transformer architectures can provide strong performance in low-resource NLP applications.
- Romanized Nepali presents unique challenges due to inconsistent transliteration and code-switching.
- Model selection should consider computational efficiency alongside predictive accuracy.
- Careful evaluation and iterative experimentation are essential components of practical machine learning development.

---

# Limitations

Several limitations provide opportunities for future investigation.

- The dataset contains only 607 labeled examples.
- Romanized Nepali lacks standardized spelling conventions.
- English-pretrained tokenization may not optimally represent Romanized Nepali vocabulary.
- Performance was evaluated using a single train-test split.
- Comparisons with multilingual transformer models were outside the scope of this study.

---

# Future Research Directions

Potential extensions include:

- Evaluate multilingual transformer models such as **mBERT** and **XLM-R**
- Compare performance against RoBERTa and DeBERTa
- Investigate transliteration normalization before tokenization
- Expand the dataset using additional annotated Romanized Nepali text
- Explore explainable AI techniques such as SHAP for model interpretation
- Study sentiment classification under code-switched Nepali-English settings

---

# Learning Outcomes

This project strengthened my understanding of:

- Transformer-based Natural Language Processing
- Transfer learning for sequence classification
- Hugging Face Transformers and PyTorch workflows
- Practical model selection under computational constraints
- Performance evaluation using precision, recall, and F1-score
- Challenges associated with low-resource language processing

---

# Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Google Colab
- Jupyter Notebook

---

# Repository Structure

```
├── NLP_Bert.ipynb
├── README.md
└── requirements.txt
```

---

# References

- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Sanh et al. (2020). *DistilBERT: A Distilled Version of BERT*
- Hugging Face Transformers Documentation
- PyTorch Documentation
