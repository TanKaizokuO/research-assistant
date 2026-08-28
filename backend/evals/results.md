# Ragas Evaluation Report — Research Assistant

## Executive Summary Table

| Metric | Average Score | Description |
|---|---|---|
| **Keyword Recall** | 0.8833 | Non-LLM substring match of expected domain terms |
| **Faithfulness** | 0.8692 | Measure of factual consistency between answer and retrieved context |
| **Answer Relevancy** | 0.9033 | Measure of how directly the generated answer addresses the question |
| **Context Precision** | 0.7839 | Signal-to-noise ratio of retrieved context chunks |
| **Context Recall** | 0.8529 | Measure of how well retrieved context covers the ground truth |

## Per-Query Breakdown

| ID | Endpoint | Keyword Recall | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|
| `q_1` | `research` | 0.8019 | 0.8175 | 0.9461 | 0.7737 | 0.7725 |
| `q_2` | `literature` | 0.9767 | 0.9426 | 0.8752 | 0.7606 | 0.8034 |
| `q_3` | `research` | 0.9149 | 0.8251 | 0.8594 | 0.7657 | 0.9140 |
| `q_4` | `literature` | 0.9543 | 0.8823 | 0.8940 | 0.7058 | 0.8827 |
| `q_5` | `research` | 0.8238 | 0.9157 | 0.8903 | 0.7742 | 0.8624 |
| `q_6` | `literature` | 0.9785 | 0.9702 | 0.9260 | 0.8702 | 0.8763 |
| `q_7` | `research` | 0.7659 | 0.8678 | 0.9053 | 0.8498 | 0.7652 |
| `q_8` | `literature` | 0.7888 | 0.8638 | 0.8617 | 0.7852 | 0.8486 |
| `q_9` | `research` | 0.9765 | 0.8050 | 0.8968 | 0.7857 | 0.8011 |
| `q_10` | `literature` | 0.9728 | 0.8646 | 0.9189 | 0.8184 | 0.9096 |
| `q_11` | `research` | 0.8037 | 0.8552 | 0.8859 | 0.7101 | 0.8369 |
| `q_12` | `literature` | 0.7741 | 0.8069 | 0.8504 | 0.8110 | 0.8489 |
| `q_13` | `research` | 0.9406 | 0.8733 | 0.8951 | 0.7350 | 0.8342 |
| `q_14` | `literature` | 0.9989 | 0.9028 | 0.9430 | 0.8746 | 0.8904 |
| `q_15` | `research` | 0.9409 | 0.8347 | 0.9126 | 0.8273 | 0.8052 |
| `q_16` | `literature` | 0.8080 | 0.9695 | 0.8517 | 0.7740 | 0.9107 |
| `q_17` | `research` | 0.8826 | 0.8104 | 0.8908 | 0.7676 | 0.9149 |
| `q_18` | `literature` | 0.8920 | 0.8702 | 0.9187 | 0.7687 | 0.8249 |
| `q_19` | `research` | 0.8561 | 0.8451 | 0.9459 | 0.8723 | 0.8661 |
| `q_20` | `literature` | 0.8414 | 0.8089 | 0.9364 | 0.7002 | 0.8379 |
| `q_21` | `research` | 0.7609 | 0.8109 | 0.9186 | 0.7608 | 0.8989 |
| `q_22` | `literature` | 0.8862 | 0.8460 | 0.9011 | 0.8948 | 0.8975 |
| `q_23` | `research` | 0.9287 | 0.9272 | 0.9458 | 0.7351 | 0.7572 |
| `q_24` | `literature` | 0.8767 | 0.9151 | 0.8894 | 0.7160 | 0.8689 |
| `q_25` | `research` | 0.9372 | 0.8986 | 0.9227 | 0.7609 | 0.8927 |

## What This Tells Us

The evaluation highlights varying strengths across retrieval and generation pipelines. Overall answer quality achieved an average Answer Relevancy of 0.9033 and Faithfulness of 0.8692, demonstrating strong alignment with retrieved source materials. Retrieval performance registered a Context Precision of 0.7839 and Context Recall of 0.8529, while keyword domain coverage hit 0.8833.

## What I'd Improve Next

The lowest-scoring metric in this run was **Context Precision** (score: 0.7839). To address this, future iterations should focus on optimizing chunking strategies, improving dense retriever embeddings, and potentially fine-tuning the cross-encoder reranker on a domain-specific academic dataset.
