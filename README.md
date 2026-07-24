This project explores fine-tuning large language models for sentiment classification under limited GPU resources. The initial goal was to fine-tune LLaMA 2 (7B Chat) on a Romanized Nepali restaurant review dataset using parameter-efficient fine-tuning techniques such as LoRA and QLoRA.

The dataset used was amirpoudel/nepal-romanized-restaurant-reviews from Hugging Face, which contains Romanized Nepali restaurant reviews labeled with three sentiment classes: positive, neutral, and negative.

The initial approach used LLaMA-2-7B-Chat with parameter-efficient fine-tuning methods, including LoRA and QLoRA. However, the model could not be loaded into GPU memory on Google Colab with 16 GB of VRAM, even when using 4-bit quantization.

Because the out-of-memory error occurred during model loading, training and evaluation could not begin. This showed that, under the available hardware constraints, LLaMA 2 was not a practical choice for this task.

The project therefore shifted to distilbert-base-uncased, which has substantially lower memory requirements and is better suited to sentence-level classification.

The final implementation successfully fine-tunes DistilBERT for sentiment classification of Romanized Nepali text, demonstrating an effective trade-off between model size, performance, and hardware constraints.

Key Highlights

Explored LoRA and QLoRA for efficient fine-tuning of large language models

Identified practical GPU memory limitations when working with 7B parameter models

Selected DistilBERT as a resource-efficient alternative for sentiment classification

Fine-tuned on a Romanized Nepali sentiment dataset

Demonstrates real-world decision-making in model selection under hardware constraints
