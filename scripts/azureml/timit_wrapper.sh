#! /usr/bin/env bash

# We use the job array for parallelism. Don't muck with these.
if [ ! -z "${RANK}" ]; then
  export TIMIT_OFFSET="${RANK}"
  export TIMIT_STRIDE="${WORLD_SIZE}"
fi

if [ -z "${db_url}" ]; then
  export db_url="$(python -c '
try:
    from azureml.core import Run

    run = Run.get_context()
    print(run.get_secret(name="db-url"))
except:
    print("")
')"
fi

bash ./timit.sh "$@" || exit 1
