# Matrix-free solvers 

Repository containing the source for the NMPDE project year 2025-26.

Authors:

- Francesco Calzona (11172853)
- Federica Censabella (10859648)
- Alessandro Masini (10940986)
- Andrea Oggioni (10822715)

## Usage

In order to compile everything locally, we provide an handy script that will istantiate the correct Apptainer containers and perform compilation inside them.

    ./compile_locally.sh

Three binaries will be created in the folder in which the script was run, namely

    ./matrix_based
    ./matrix_free_no_simd
    ./matrix_free_simd

This script needs to know where to find the `amsc_mk_2025.sif` and the `dealii-avx512.sif` containers: see `./compile_locally.sh --help` for more details.

The first one can be downloaded from quay.io while the second one must be built locally with the provided dockerfile.

    apptainer pull https://quay.io/repository/pjbaioni/amsc_mk:2025
    docker build -t dealii-avx512:latest .
    apptainer build my-app.sif docker-daemon://dealii-avx512:latest

Be aware that compiling the docker image may take several minutes as multiple libraries needs to be compiled.

PBS job scripts to perform extensive tests or to launch the binaries in the same conditions we did to perform our benchmarks are provided into `pbs_jobs/`.

The `run_extensive_tests.sh` script is used by those jobs to run tests with given parameters but it can also be used manually and needs to know where to find the two aforementioned containers: see `./run_extensive_tests.sh --help` for more details.

It is possible to build the documentation locally, just run `doxygen`. The same documentation, updated to the last push to main, is available at [alessandromasini.github.io](https://alessandromasini.github.io/nmpde-projects-matrix-free-solvers/).

## Credits

This work is based on the [Deal.II library]() with its tutorials [step-9](https://dealii.org/current/doxygen/deal.II/step_9.html), [step-37](https://dealii.org/current/doxygen/deal.II/step_37.html), [step-40](https://dealii.org/current/doxygen/deal.II/step_40.html) and [step-48](https://dealii.org/current/doxygen/deal.II/step_48.html).
