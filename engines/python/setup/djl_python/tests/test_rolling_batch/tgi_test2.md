# TGI test
model="TheBloke/Llama-2-13B-Chat-fp16"
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:0.9.4 \
--model-id $model --num-shard 4

tgi==0.9.4
transformers==4.29.2

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":50,"do_sample":false, "repetition_penalty":1.0}}' \
    -H 'Content-Type: application/json'
```
"\n\nHere is a simple program to add two numbers in Python:\n```\na = 5\nb = 3\n\nsum = a + b\n\nprint(\"The sum is:\", sum)\n```\nThis program will"

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Deep mind is a","parameters":{"max_new_tokens":50,"do_sample":false, "repetition_penalty":1.0}}' \
    -H 'Content-Type: application/json'
```
"term used to describe the collective unconscious of humanity, a shared reservoir of archetypes and memories that underlie our individual experiences and dreams. It is a concept developed by Carl Jung, a Swiss psychiatrist"

```bash
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Memories follow me left and right. I can","parameters":{"max_new_tokens":50,"do_sample":false, "repetition_penalty":1.0}}' \
    -H 'Content-Type: application/json'
```
"'t seem to shake them off. They are like a shadow that haunts me wherever I go.\n\nI try to distract myself with work, with hobbies, with anything that can take my mind off of them."
