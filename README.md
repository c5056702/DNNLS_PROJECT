# DNN Project

## Graph-Based Entity Reasoning for Visual Story Continuation

Purpose

- Improve multimodal story continuation by modeling entity relations across time and fusing graph signals into sequence prediction.

Method overview

- Image context: ResNet18 embeddings (normalized) for each context frame.
- Text context: TF-IDF vectors projected to 512-dim embeddings.
- Sequence model: GRU over fused image+text embeddings to predict next-step image/text embeddings.
- Graph reasoning: entity extraction from `<gdo ...>` tags, graph construction (co-occurrence + temporal proximity), gated message passing, and graph pooling.
- Fusion variants: full graph fusion and a text-only graph fusion option.

Repository layout

- `src/dataloader.py`: windowed dataset, TF-IDF collators, graph construction and padding.
- `src/encoders_image.py`: ResNet18 embedder.
- `src/encoders_text.py`: TF-IDF vectorizer utilities and text cleaning.
- `src/model_baseline.py`: ResNet+TF-IDF GRU baseline.
- `src/graph_module.py`: simple gated graph reasoner.
- `src/model_graph.py`: graph-fused predictors.
- `src/train.py`: training loop for baseline/graph modes.
- `src/eval.py`: validation metrics helpers.
- `src/download_dataset.py`: Hugging Face StoryReasoning download/inspection.
- `src/prepare_manifest.py`: index preparation for K-step windows.

Quickstart

```bash

cd "Mani"

# Baseline (ResNet + TF-IDF)
python src/train.py --mode baseline

# Graph-fused
python src/train.py --mode graph

# Graph fused into text head only
python src/train.py --mode graph_textonly
```

Notes

- The graph pipeline expects entity tags in the text like `<gdo charX>NAME</gdo>` to extract entities reliably

Project 1: Entity-Graph Reasoning for Multimodal Story Continuation

Project motivation

Multimodal storytelling requires understanding not only visual content but also the narrative structure that connects events across time. In stories, entities such as characters and objects persist and interact over multiple frames. Standard sequence models process multimodal inputs implicitly and often fail to maintain long-term narrative coherence, especially in text generation. This project investigates whether explicitly modeling entities and their relationships using graph reasoning can improve multimodal story continuation.

Task definition

The task is to predict the next step in a multimodal story. Given a temporal context of K=4 previous frames, where each frame contains an image and associated story text, the model must predict the embedding of the next image and the embedding of the next piece of text. The objective is to improve temporal and narrative coherence in both modalities.

Dataset summary (StoryReasoning)

| Item | Value |
| --- | --- |
| Source | `daniel3303/StoryReasoning` (Hugging Face) |
| Splits | Train: 3,552 rows, Test: 626 rows |
| Schema | `story_id`, `images`, `frame_count`, `chain_of_thought`, `story` |
| Missing-rate check | ~0.0 for all columns (sampled 1,000 rows per split) |

Qualitative inspection

- Stories include markup tags like `<gdi imageN>` blocks and `<gdo ...>` entity/object spans.
- `chain_of_thought` is structured with sections per image, including tables for Characters, Objects, and Setting.

EDA performed (plots generated in Colab)

- Frame count distribution (train/test hist + boxplot).
- Token length histograms for `story` and `chain_of_thought` (sampled 1,200 rows).
- Scatter of `frame_count` vs story token length (sampled 1,200 rows).
- Keyword frequency for temporal/entity cues (sampled 1,500 rows) for: then, after, before, suddenly, later, while, because, he, she, they.
- Heuristic entity mining via capitalized tokens in `chain_of_thought` (sampled 800 rows).

Windowing + splits

- Train/val split by `story_id` with 10% validation: 3,196 train stories, 356 val stories, 626 test stories.
- Frame-text alignment: marker-based split failed, so story text is split into sentences and evenly assigned across `frame_count` frames.
- Sliding window size K=4 (predict frame t from frames t-4..t-1).
- Sample counts after windowing: train 27,107; val 2,884; test 5,254.

Data processing

- Images: ResNet18 pretrained on ImageNet, last layer removed, 512-d normalized embeddings, frozen encoder.
- Text: remove annotation tags, clean whitespace, TF-IDF with 8,000 features, projected to 512-d with a trainable linear layer.

Baseline model

- Concatenate image and text embeddings per timestep, project, then GRU over time.
- Final hidden state predicts next image and next text embeddings via separate heads.
- Training uses cosine similarity loss for both modalities.

Entity graph construction

- Nodes represent unique entities from annotated text in the previous K frames.
- Node features: entity type, frequency, first/last appearance, TF-IDF of entity name.
- Edges: entity co-occurrence within the same frame and temporal overlap across frames.

Graph neural network

- Lightweight message passing with normalized adjacency and gated updates.
- Pooled node representations produce a single graph embedding for the temporal window.

Model variants

| Variant | Graph usage | Fusion target |
| --- | --- | --- |
| Baseline | None | N/A |
| Graph-fused | Graph embedding | Shared temporal hidden state |
| Graph text-only | Graph embedding | Text prediction head only |

Experimental results (validation cosine similarity)

| Model | Image cos | Text cos |
| --- | --- | --- |
| Baseline | 0.8023 | 0.8397 |
| Graph-fused | 0.7904 | 0.8916 |
| Graph text-only | 0.8039 | 0.8149 |

Key findings

- Explicit entity graph reasoning substantially improves text-side narrative coherence.
- Best gains occur when graph embedding is fused into the shared temporal state.
- Aggressive fusion can slightly reduce image prediction performance since the graph is derived from text.
- Fusion strategy is critical to balance multimodal performance.
- Early stopping is important; best results occur in the first epoch, with later epochs overfitting.

Conclusion

Explicit modeling of entities and their relationships via graph reasoning improves multimodal story understanding, especially for language coherence. While graph reasoning can introduce trade-offs across modalities, careful architectural design can control these effects. The results support the hypothesis that structured reasoning modules complement sequence models in multimodal narrative tasks.
