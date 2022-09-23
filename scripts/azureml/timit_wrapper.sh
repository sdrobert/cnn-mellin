#! /usr/bin/env bash

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
