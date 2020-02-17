#!/bin/sh
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "autogen: TCCL"
$DIR/tccl/autogen.sh

echo "autogen: MCCL"
rm -rf autom4te.cache
mkdir -p config/m4 config/aux
autoreconf -f --install || exit 1
rm -rf autom4te.cache
exit 0
