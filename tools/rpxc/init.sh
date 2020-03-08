#!/usr/bin/env bash

set -e

readonly script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export RPXC_IMAGE=rpxc-rust

sha1sum "${script_dir}/Dockerfile" > "${script_dir}/.Dockerfile.sha1"

docker build -t "${RPXC_IMAGE}" "${script_dir}"

docker run ${RPXC_IMAGE} > "${script_dir}/.rpxc.sh"

sed -i 's|#!/bin/bash|#!/bin/bash\nRPXC_IMAGE=rpxc-rust|' "${script_dir}/.rpxc.sh"

chmod +x "${script_dir}/.rpxc.sh"
