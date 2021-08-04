#!/bin/bash

docker-compose run --rm -d --service-ports tensorboard \
 && echo "-- Tensorboard started in detached mode --";
