# Obtaining the corpora

All corpora are downloaded locally into `data/` (gitignored). After placing the
files, run `make data` to build the manifests and `reports/data_card.md`.

## LibriSpeech (required)

Source: https://www.openslr.org/12 · License: CC BY 4.0

Download and extract the two clean splits so the layout is
`data/LibriSpeech/dev-clean/...` and `data/LibriSpeech/test-clean/...`:

    cd data
    curl -O https://www.openslr.org/resources/12/dev-clean.tar.gz
    curl -O https://www.openslr.org/resources/12/test-clean.tar.gz
    tar xzf dev-clean.tar.gz
    tar xzf test-clean.tar.gz

Expected archive sizes: dev-clean 337,926,286 bytes, test-clean 346,663,984
bytes. Verify the archive is complete before extracting (`gzip -t <file>`).

## AMI meeting corpus

Audio and annotations: https://groups.inf.ed.ac.uk/ami/corpus/ · License: CC BY 4.0

Place the ground-truth RTTMs under `data/ami/rttms/<meeting>.rttm` and the
abstractive summaries under `data/ami/abstractive/<meeting>.abssumm.xml`. The
summaries come from the manual annotation archive
`ami_public_manual_1.6.2.zip` (the `abstractive/` folder inside it). The meetings
used are configured in `configs/data.yaml`; see ADR 0003 for the subset.

## Accented English (optional, used in the robustness phase)

Common Voice accent-labelled English requires accepting the publisher's terms on
the dataset page before download. Place a TSV with `path`, `sentence`,
`accents` and `client_id` columns at `data/accents/validated.tsv` and the audio
clips under `data/accents/clips/`. Per-accent WER is reported only once these
files are present; groups with fewer than 100 clips are reported, not concluded
from.

## Consent

This tool must not be run on a recording of a real meeting without the consent
of everyone recorded.
