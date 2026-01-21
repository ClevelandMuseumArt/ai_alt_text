### Cheat Sheet Execution Commands

#### test_gpu.py
Checks device capabilities for efficient script running:
```shell
python -m test_gpu
```

#### artwork_bulk_load.py
Download Current Collection Online Artwork Data:
```shell
python -m artwork_bulk_load
```

#### generate_alt_text.py
Gemini 3 Model with both RAG and single shot prompt, storing metrics:
```shell
python -m generate_alt_text --bulk --bulk-data-path <custom csv file path here> --gemini-credentials-file gemini-key.json --gemini-model gemini-3-pro-preview --prompt-file prompt.md --rag-directory rag_examples --with-rag --store-metrics --log-level DEBUG
```

#### analyze_results.py
Run data analysis on output csv files
```shell
python -m analyze_results output.just results_1.csv results_2.csv
```
