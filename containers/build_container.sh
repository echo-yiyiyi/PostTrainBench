#!/bin/bash
module load singularity

container="${1}"

export POST_TRAIN_BENCH_CONTAINERS_DIR=${POST_TRAIN_BENCH_CONTAINERS_DIR:-containers}
export APPTAINER_BIND=""

export SINGULARITY_TMPDIR=/ibex/user/$USER/tmpdir
export SINGULARITY_CACHEDIR=/ibex/user/$USER/cachedir

singularity build --fakeroot "${POST_TRAIN_BENCH_CONTAINERS_DIR}/${container}.sif" "containers/${container}.def"
