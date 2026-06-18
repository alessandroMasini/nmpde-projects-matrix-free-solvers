# Matrix-free solvers 

Repository containing the source for the NMPDE project year 2025-26.

Authors:

- Francesco Calzona (11172853)
- Federica Censabella (10859648)
- Alessandro Masini (10940986)
- Andrea Oggioni (10822715)

## Usage

This project requires two Apptainer containers (namely `amsc_mk_2025.sif` and `dealii-avx512.sif`) to be available on the machine. In order to obtain those containers, run

    apptainer pull https://quay.io/pjbaioni/amsc_mk:2025
    docker build -t dealii-avx512:latest .
    apptainer build dealii-avx512.sif docker-daemon://dealii-avx512:latest

Be aware that compiling the Docker image may take several minutes (also hours, depending on the machine) as multiple libraries need to be compiled. Be also aware that you will need an avx512 enabled processor to do that. If the Docker image does not compile succesfully, you can still run the non-simd variant of our implementation.

Alternatively, you can download `dealii-avix512.sif` from our [onedrive](https://polimi365-my.sharepoint.com/:u:/g/personal/10822715_polimi_it/IQD8itF4oDDaRLfxLngnEo7NAXkie_eyf5viK1ILk6a6Xig?e=PAItp5).

In order to compile our solvers locally, we provide an handy script that will instantiate the correct Apptainer containers and perform compilation inside them.

    ./compile_locally.sh

This script needs to know where to find the `amsc_mk_2025.sif` and the `dealii-avx512.sif` containers: see `./compile_locally.sh --help` for more details.

Three binaries will be created in the folder in which the script was run, namely

    ./matrix_based  # requires amsc_mk_2025.sif
    ./matrix_free_no_simd  # requires amsc_mk_2025.sif
    ./matrix_free_simd  # requires dealii-avx512.sif

PBS job scripts to perform extensive tests or to launch the binaries in the same conditions we did to perform our benchmarks are provided into `pbs_jobs/`.

The `run_extensive_tests.sh` script is used by those jobs to run tests with given parameters, but it can also be used manually and needs to know where to find the two aforementioned containers: see `./run_extensive_tests.sh --help` for more details.

It is possible to build the documentation locally, just run `doxygen`. The same documentation, updated to the last push to main, is available at [alessandromasini.github.io](https://alessandromasini.github.io/nmpde-projects-matrix-free-solvers/).

## Credits

This work is based on the [Deal.II library]() with its tutorials [step-9](https://dealii.org/current/doxygen/deal.II/step_9.html), [step-37](https://dealii.org/current/doxygen/deal.II/step_37.html), [step-40](https://dealii.org/current/doxygen/deal.II/step_40.html) and [step-48](https://dealii.org/current/doxygen/deal.II/step_48.html).
