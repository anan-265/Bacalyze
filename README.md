# Unnamed

## Licensing Information

This software is licensed under the **GNU Affero General Public License v3 (AGPL v3)**.

## Dependencies

This software has the following dependencies, each of which is governed by its own license. Users are responsible for ensuring compliance with each dependency's license.

### System Dependencies:
- **Python**: Licensed under the [Python Software Foundation License](https://docs.python.org/3/license.html)
- **BWA**: Licensed under the [GNU GPL License v3](https://github.com/lh3/bwa?tab=GPL-3.0-1-ov-file)
- **Fastp**: Licensed under the [MIT License](https://github.com/OpenGene/fastp/blob/master/LICENSE)
- **SAMtools**: Licensed under the [MIT License](https://github.com/samtools/samtools/blob/develop/LICENSE)
- **BCFtools**: Licensed under the [MIT License](https://github.com/samtools/bcftools/blob/develop/LICENSE)
- **VCFtools**: Licensed under the [GNU Lesser GPL License v3](https://github.com/vcftools/vcftools/blob/master/LICENSE)
- **Spades**: Licensed under the [GNU GPL License v3](https://github.com/ablab/spades?tab=License-1-ov-file)
- **Prokka**: Licensed under the [GNU GPL License v3](https://raw.githubusercontent.com/tseemann/prokka/master/doc/LICENSE.Prokka)
- **MEFinder**: Licensed under the [GNU GPL License v3](https://pypi.org/search/?c=License+%3A%3A+OSI+Approved+%3A%3A+GNU+General+Public+License+v3+%28GPLv3%29)
- **Resfinder**: Licensed under the [Apache License 2.0](https://github.com/genomicepidemiology/resfinder?tab=License-1-ov-file)
- **Nextflow**: Licensed under the [Apache License 2.0](https://github.com/nextflow-io/nextflow/tree/master?tab=Apache-2.0-1-ov-file)

### Python Packages:
- **pandas**: Licensed under the [BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/main/LICENSE)
- **scikit-learn**: Licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING)
- **Biopython**: Licensed under the [BSD 3-Clause License](https://github.com/biopython/biopython?tab=License-2-ov-file)

## Docker Information

This software requires **Docker** to build and run. Docker must be installed separately.

- Docker is licensed separately by [Docker, Inc.](https://www.docker.com/).
- Users are responsible for ensuring compliance with Docker's licensing terms.
- For more details, refer to [Docker's licensing page](https://docs.docker.com/subscription/desktop-license/).

## Installation

1. Install **Docker** by following the instructions on the [Docker website](https://www.docker.com/get-started).
2. Clone this repository to your local machine.
3. Build the Docker image by running the following command:
   
   ```bash
   docker build -t kani .
