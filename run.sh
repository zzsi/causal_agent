#!/bin/sh
# Run the causal agent container

docker run --rm -it -e OPENAI_API_KEY="${OPENAI_API_KEY}" causal_agent "$@"
