#!/bin/bash

docker-compose -f docker-compose.yml -f docker-compose.prod.yml run --service-ports --rm server
