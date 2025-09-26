#!/usr/bin/env sh
set -e
host="$1"; shift
cmd="$@"
until nc -z $host 80; do
  echo "waiting for $host..."; sleep 1;
done
exec $cmd
