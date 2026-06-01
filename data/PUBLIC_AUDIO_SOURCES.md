# Public Mandarin Audio Sources

This project uses only public audio sources with recorded provenance. Numeric model results must be generated from actual downloaded files and metadata.

## Primary candidate: THCHS-30 / OpenSLR SLR18

- Source page: https://www.openslr.org/18/
- Dataset name: THCHS-30, Tsinghua Continuous Mandarin Speech Database
- Language: Mandarin Chinese
- License/terms stated on source page: Apache License v2.0; free for academic users
- Downloaded files:
  - `data_thchs30.tgz` from `https://openslr.trmal.net/resources/18/data_thchs30.tgz`
  - optional `resource.tgz` from `https://openslr.trmal.net/resources/18/resource.tgz`
- Suitability: includes Mandarin speech and transcripts. Tone labels are not provided directly, so tone labels must be derived from transcript pinyin/lexicon resources or a documented conversion step.

## Candidate: AISHELL-1 / OpenSLR SLR33

- Source page: https://www.openslr.org/33/
- Dataset name: AISHELL-1
- Language: Mandarin Chinese
- License/terms stated on source page: Apache License v2.0; free for academic use
- Download links listed by source:
  - `https://openslr.trmal.net/resources/33/data_aishell.tgz`
  - `https://openslr.trmal.net/resources/33/resource_aishell.tgz`
- Suitability: large 16 kHz Mandarin corpus with transcripts and speaker metadata; useful for a larger model comparison but heavier to download and process.

## Candidate: Free ST Chinese Mandarin Corpus / OpenSLR SLR38

- Source page: https://www.openslr.org/38/
- Dataset name: Free ST Chinese Mandarin Corpus
- Language: Mandarin Chinese
- License/terms stated on source page: Creative Commons BY-NC-ND 4.0
- Download link listed by source:
  - `https://openslr.trmal.net/resources/38/ST-CMDS-20170001_1-OS.tar.gz`
- Suitability: Mandarin speech with transcripts from many speakers. License is more restrictive than Apache-licensed sources, so use carefully and cite terms.

## GitHub search note

GitHub repository search for small Mandarin tone audio datasets did not return a clearly usable repository with downloadable audio and clear license during the initial search. OpenSLR sources above are therefore preferred for reproducibility and citation.
