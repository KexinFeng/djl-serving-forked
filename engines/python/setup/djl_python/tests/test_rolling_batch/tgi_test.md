# TGI test
model="huggyllama/llama-7b"

## lmi_dit testing
deepjavalibrary/djl-serving:deepspeed-nightly
transformers==4.29.2

## version 1
tgi==0.9.4
transformers==4.29.2

docker run --gpus all --shm-size 1g -p 8088:88 -v $volume:/data ghcr.io/huggingface/text-generation-inference:0.9.4 --model-id $model


curl 127.0.0.1:8088/generate \
    -X POST \
    -d '{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'

\nwrite a program to add two numbers in python.\nWrite a program to add two numbers in python.\nThe program should ask the user to enter two numbers and then add them.\nThe program should print the sum of the two numbers.
xxxxxx

curl 127.0.0.1:8088/generate \
    -X POST \
    -d '{"inputs":"Deep mind is a","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'

British artificial intelligence company that was founded in 2014. It is a subsidiary of Google. It is based in London, UK. It is a part of Google’s parent company Alphabet.\nDeep mind is a
xxxxxxx

curl 127.0.0.1:8088/generate \
    -X POST \
    -d '{"inputs":"Memories follow me left and right. I can","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'

’t escape them. I can’t escape the pain. I can’t escape the guilt. I can’t escape the fear. I can’t escape the anger. I can’t escape the sadness. I can’
xxxxxxx

## version2
tgi==1.0.2
transformers==4.32.0

model="huggyllama/llama-7b"
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:1.0.2 --model-id $model

curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'

\nwrite a program to add two numbers in python.\nWrite a program to add two numbers in python.\nThe program should ask the user to enter two numbers and then add them.\nThe program should print the sum of the two numbers.

curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Deep mind is a","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'

British artificial intelligence company that was founded in 2014. It is a subsidiary of Google. It is based in London, UK. It is a part of Google’s parent company Alphabet.\nDeep mind is a


curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Memories follow me left and right. I can","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'

’t escape them. I can’t escape the pain. I can’t escape the guilt. I can’t escape the fear. I can’t escape the anger. I can’t escape the sadness. I can’


## version3
tgi==0.9.2
transformers==4.29.2

model="huggyllama/llama-7b"
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:0.9.0 --model-id $model

curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"Memories follow me left and right. I can","parameters":{"max_new_tokens":50}}' \
    -H 'Content-Type: application/json'
