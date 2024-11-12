# --- Licensing and Dependency Information ---
# This software is licensed under the GNU Affero General Public License v3 (AGPL v3).
# Dependencies:
# - Python: Licensed under the Python Software Foundation License.
# - BWA: Licensed under the GPL License.
# - Fastp: Licensed under the MIT License.
# - SAMtools: Licensed under the MIT License.
# - BCFtools: Licensed under the MIT License.
# - VCFtools: Licensed under the GPL License.
# - Nextflow: Licensed under the Apache License 2.0.
# - pandas: Licensed under the BSD 3-Clause License.
# - scikit-learn: Licensed under the BSD 3-Clause License.
# 
# Users are responsible for complying with the licenses of the dependencies.
# For more details, see the relevant license files for each tool.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN echo -e "Licensing Information:\n\
This software is licensed under the GNU Affero General Public License v3 (AGPL v3).\n\
Dependencies and their respective licenses:\n\
- Python: Python Software Foundation License\n\
- BWA: GPL License\n\
- Fastp: MIT License\n\
- SAMtools: MIT License\n\
- BCFtools: MIT License\n\
- VCFtools: GPL License\n\
- Nextflow: Apache License 2.0\n\
- pandas: BSD 3-Clause License\n\
- scikit-learn: BSD 3-Clause License\n\
Users are responsible for ensuring compliance with each dependency's license." && \
	apt-get update && \
    apt-get install -y \
    python3 python3-pip bwa fastp samtools bcftools vcftools curl openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/* && \
    # Download and install Nextflow
    curl -s https://get.nextflow.io | bash && \
    mv nextflow /usr/local/bin/ && \
    # Use pip to install Python packages, breaking system package management
    pip3 install --break-system-packages pandas scikit-learn && \
    # Set Python3 as the default python command
    ln -s /usr/bin/python3 /usr/bin/python

CMD ["bash"]