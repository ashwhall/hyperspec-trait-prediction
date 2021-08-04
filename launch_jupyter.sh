#!/bin/bash

docker-compose run --rm -d --service-ports jupyter \
 && echo "-- Jupyter started in detached mode --";
