#!/bin/sh
# Run the causal agent container

ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE file" >&2
  exit 1
fi

docker run --rm -it -v "$(pwd)/$ENV_FILE:/app/.env" causal_agent "$@"
