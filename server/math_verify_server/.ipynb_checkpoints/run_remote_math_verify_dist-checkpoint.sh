

#!/bin/bash

set -x

# python ./python_compile.py \
#     --port ${PORT}

export BIND=0.0.0.0:${PORT}

gunicorn -b ${BIND} -c code_conf.py math_verify_server:app
sleep 365d