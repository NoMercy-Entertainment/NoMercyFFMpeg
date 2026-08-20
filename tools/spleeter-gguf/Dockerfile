# tools/spleeter-gguf/Dockerfile
FROM python:3.11-slim

# sentencepiece: gguf==0.10.0's gguf/vocab.py unconditionally does
# `from sentencepiece import SentencePieceProcessor` at import time, even
# though sentencepiece is not declared as one of the package's own
# dependencies (only numpy, pyyaml, tqdm are). Without it, `import gguf`
# raises ModuleNotFoundError before any of our code runs. Not in the brief;
# found while building this image.
RUN pip install --no-cache-dir tensorflow-cpu==2.16.2 numpy==1.26.4 gguf==0.10.0 sentencepiece

WORKDIR /work
COPY convert.py dump_reference.py compare.py /work/

# The 2stems checkpoint, MIT-licensed, from the spleeter release assets.
ADD https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz /work/2stems.tar.gz
RUN mkdir -p /work/2stems && tar -xzf 2stems.tar.gz -C /work/2stems && rm 2stems.tar.gz

ENTRYPOINT ["python"]
