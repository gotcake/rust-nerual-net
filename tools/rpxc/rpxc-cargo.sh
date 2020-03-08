#!/usr/bin/env bash

set -e

readonly script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ ! -f "${script_dir}/.rpxc.sh" ]] || [[ "0" != "$(sha1sum -c "${script_dir}/.Dockerfile.sha1" > /dev/null 2>&1; echo $?)" ]]; then
    "${script_dir}/init.sh"
fi

cd "${script_dir}/../../"

readonly first_arg=$1
shift

"${script_dir}/.rpxc.sh" cargo "${first_arg}" --target=armv7-unknown-linux-gnueabihf $@