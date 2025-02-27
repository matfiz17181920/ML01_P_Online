## How to pull image
docker image pull tohakratok/model_titanic_server

## How to run image
docker run -d  --rm -p  5000:5000 -v ${pwd}/html:/www -v  ${pwd}/logs:/var/log/nginx --name titanicserver tohakratok/model_titanic_server:latest