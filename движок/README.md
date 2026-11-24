# Движок поиска по памяти

Облачный расчёт на GitHub Actions (16 ГБ RAM, 4 CPU). Песочница агента 2 ГБ не тянет bge-m3 — облако тянет.

## Состав
- `engine.py` — FTS5 (BM25, rank) + LIKE + леммы (pymorphy3) + RRF k=60 + реранк bge-reranker-v2-m3-en-ru ONNX int8 + векторная рука bge-m3.
- `build.yml` — workflow: сборка, замер MRR, поиск.

## Источники (free-first, всё проверено)
| Что | Откуда |
|---|---|
| FTS5 rank быстрее bm25 | sqlite.org/fts5 §5.2 |
| RRF k=60 | MariaDB docs, Elastic RRF |
| реранк ONNX int8 | metalmon80/bge-reranker-v2-m3-en-ru-onnx |
| +20 п.п. Hit@1 от реранка | aimultiple.com/rerankers |
| RusBEIR: BM25 36,7 → BGE-M3 57,9 → +реранк 69,1 nDCG@10 | arXiv 2504.12879 |
| bge-m3 на CPU: 1,6 ГБ RAM, 0 VRAM | haotianblog /local-llm-deploy-04-rag |

## Запуск
Actions → memory-engine-build → Run workflow → step=build.

## Замеры
- индекс: 2167 записей → 48 996 кусков, 278 МБ, 36 с (локально)
- MRR после 6 фиксов: 1,000 / 0,975 / 1,000 / 0,148
