# Indic Translation

As the name conveys, this repository contains code to translate Indic languages (Hindi, for now) to English.

## Install dependencies

### huggingface/transformers
We use the Transformer (mostly Bert and its flavours) models provided by [huggingface](https://github.com/huggingface/transformers). Build the package from source. For more details on the dependencies for huggingface/transformers, please follow the README.md on their repository.

```bash
if [ ! -d "transformers" ]; then git clone https://github.com/huggingface/transformers; fi
cd transformers
pip install .
```

### BPEmb
BPEmb is a collection of pre-trained subword embeddings which can be installed by

```bash
pip install bpemb
```
For more details on the usage of the package, refer [BPEmb](https://github.com/bheinzerling/bpemb).

## Usage
## Contributing
## License
