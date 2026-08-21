# Dataset and benchmark attribution

This repository contains code and derived research outputs. It does not
redistribute the raw corpora or adapter weights. Users are responsible for
reviewing the current terms of each source before downloading or reusing data.

## Adapter training corpus

### PANDORA Big Five redistribution

- Repository: [`jingjietan/pandora-big5`](https://huggingface.co/datasets/jingjietan/pandora-big5)
- Use here: all 3,006,566 rows form the training corpus for the 32 OCEAN LoRA
  adapters.
- Repository license metadata: Apache-2.0, as displayed by Hugging Face.
- Upstream work: Matej Gjurković, Vanja Mladen Karan, Iva Vukojević,
  Mihaela Bošnjak, and Jan Snajder. 2021. "PANDORA Talks: Personality and
  Demographics on Reddit." *Proceedings of the Ninth Workshop on NLP for Social
  Media*, pages 138-152. DOI:
  [`10.18653/v1/2021.socialnlp-1.12`](https://doi.org/10.18653/v1/2021.socialnlp-1.12).
- Important distinction: the Apache-2.0 label above is the metadata on the
  Hugging Face redistribution. It should not be read as an independent claim
  about every underlying Reddit post or the original request-gated release.

## Behavioral benchmark

### IPIP-NEO-120

- Canonical item and key page:
  [International Personality Item Pool](https://ipip.ori.org/30FacetNEO-PI-RItems.htm)
- Use here: 120 exact item statements and their positive or negative scoring
  keys in the frozen behavioral pilot.
- Status: the IPIP-NEO-120 is a public-domain inventory.
- Citation: John A. Johnson. 2014. "Measuring thirty facets of the Five Factor
  Model with a 120-item public domain inventory: Development of the
  IPIP-NEO-120." *Journal of Research in Personality* 51:78-89. DOI:
  [`10.1016/j.jrp.2014.05.003`](https://doi.org/10.1016/j.jrp.2014.05.003).

## Other ingestion sources

These sources are referenced by the broader dataset-construction pipeline but
were not used to train the 32 OCEAN adapters analyzed in the corrected study.

### Kaggle MBTI redistribution

- Repository: [`jingjietan/kaggle-mbti`](https://huggingface.co/datasets/jingjietan/kaggle-mbti)
- Use here: optional MBTI corpus ingestion and cleaning.
- Repository license metadata: Apache-2.0.
- Hugging Face DOI: [`10.57967/hf/3955`](https://doi.org/10.57967/hf/3955).
- Citation supplied by the dataset card: Jing Jie Tan, Ban-Hoe Kwan, Danny
  Wee-Kiat Ng, and Yan-Chai Hum. 2025. "Adaptive focal loss with personality
  stratification for stably mitigating hard class imbalance in
  multi-dimensional personality recognition." *Scientific Reports* 15.
  DOI: [`10.1038/s41598-025-22853-y`](https://doi.org/10.1038/s41598-025-22853-y).

### Reddit MBTI redistribution

- Repository: [`minhaozhang/mbti`](https://huggingface.co/datasets/minhaozhang/mbti)
- Use here: optional MBTI corpus ingestion and cleaning.
- Provenance limitation: the dataset card does not provide a verified license,
  paper citation, or collection statement. This repository therefore makes no
  license claim for that dataset.

### Essays Big Five redistribution

- Repository: [`jingjietan/essays-big5`](https://huggingface.co/datasets/jingjietan/essays-big5)
- Use here: downloaded by an importer but not used in the 32-adapter study.
- Repository license metadata: Apache-2.0.
- Hugging Face DOI: [`10.57967/hf/3956`](https://doi.org/10.57967/hf/3956).
- This repository does not redistribute the dataset.
