nextflow.enable.dsl=2

params.ref = ''
params.read1 = ''
params.read2 = ''
params.reads = ''
params.qual = '15'
params.pca = ''
params.model = ''
params.bed = ''
params.outdir = './'
params.mode = ''
params.labels = ''
params.k = '5'
params.annotation = 'no'
params.species = ''
params.fasta_reads = ''

process indexReference {
    input:
    path reference
    output:
    path "${reference}*"
    script:
    """
    bwa index -p $reference $reference
    echo "Indexing reference"
    """
}

//process trimming {
//    input:

//    path raw1
//    path raw2
//    output:
//    tuple path("${raw1}_trimmed.fastq"), path("${raw2}_trimmed.fastq")
//    script:
//   """
//   echo "Trimming raw data. Inputs: ${raw1} and ${raw2} Quality: ${params.qual}"
//   fastp -i $raw1 -I $raw2 -o ${raw1}_trimmed.fastq -O ${raw2}_trimmed.fastq -q ${params.qual}
//   """
//

process trimming {
publishDir params.outdir, mode: 'copy'
    input:
    tuple path(raw1), path(raw2)
    output:
    tuple path("${raw1}_trimmed.fastq"), path("${raw2}_trimmed.fastq")
    script:
    """
    echo "Trimming raw data. Inputs: ${raw1} and ${raw2} Quality: ${params.qual}"
    fastp -i $raw1 -I $raw2 -o ${raw1}_trimmed.fastq -O ${raw2}_trimmed.fastq -q ${params.qual}
    """
}

process trimming_single_end {
publishDir params.outdir, mode: 'copy'
    input:
    path raw1
    output:
    path "${raw1}_trimmed.fastq"
    script:
    """
    fastp -i $raw1 -o ${raw1}_trimmed.fastq -q ${params.qual}
    """
}

process alignReads {
    input:
    path ref_index
	tuple path(forward), path(reverse)
    path reference
	
    output:
    path "*.sam"

    script:
        """
	echo "Aligning Reads. Reference: ${reference}"
	bwa mem -p $reference $forward $reverse > ${forward}.sam
        """
}

process alignReads_single_end {
publishDir params.outdir, mode: 'copy'
    input:
    path ref_index
    path read1
    path reference
    
    output:
    path "*.sam"
    script:
    """
    echo "Aligning Reads. Reference: ${reference}"
    bwa mem -p $reference $read1 > ${read1}.sam
    """
}

process samToBam {
publishDir params.outdir, mode: 'copy'
    input:
    path sam
    output:
    path "*.bam"
    script:
    """
    samtools view -Sb $sam > ${sam}.bam
    """
}

process sorting {
publishDir params.outdir, mode: 'copy'
    input:
    path bam
    output:
    path "*.bam"
    script:
    """
    samtools sort $bam -o ${bam}_sorted.bam
    """
}

process variantCalling {
publishDir params.outdir, mode: 'copy'
    input:
    path sorted
    path reference
    output:
    path "${sorted}*"
    script:
    """
    bcftools mpileup -f $reference $sorted | bcftools call -vm -Oz > ${sorted}.vcf.gz
    """
}

process mutationAnnotation {
    publishDir params.outdir, mode: 'copy'
	input:
    path vcf
    path bed
    output:
    path "*.vcf"
    script:
    """
	bcftools annotate -a $bed -h <(echo -e "##INFO=<ID=ANN,Number=1,Type=String,Description=\"Annotation\">") -c CHROM,FROM,TO,INFO/ANN $vcf -o ${vcf}_annotated.vcf
    """
}

process SNP {
    input:
    path variant
    output:
    path "${variant}*"
    script:
    """
    echo "${variant}"
    vcftools --gzvcf $variant --minDP 4 --max-missing 0.2 --minQ 30 --recode --recode-INFO-all --out ${variant}.filter
    """
}

process SNPFiltering {
    input:
    path snp
    output:
    path "${snp}*"
    script:
    """
    vcftools --gzvcf $snp --remove-indels --recode --recode-INFO-all --out ${snp}.filter.snps
    """
}

process SNPOutput {
    input:
    path snps
    
    output:
    path "${snps}*"
    
    script:
    """
    bcftools query -f '%CHROM %POS %REF %ALT [%TGT]\n' $snps -o ${snps}.filter.snps.extract.txt
    """
}

process posRefAlt {
    input:
    path snpout
    output:
    path "${snpout}*"
    script:
    """
    cut -d " " -f 2,4 $snpout > ${snpout}.snp
    """
}

process encode {
	publishDir params.outdir, mode: 'copy'
    input:
    path snp
    path pca
    output:
    path "*.csv"
    script:
    """
    python3 -c "
    import pandas as pd
    import pickle
    import random
    import numpy as np
    from sklearn.decomposition import PCA
    #pd.set_option('future.no_silent_downcasting', True)
    df = pd.read_csv('${snp}', sep=' ', header=None)
    df_transposed = df.transpose()
    new_header = df_transposed.iloc[0]
    df_transposed = df_transposed[1:]
    df_transposed.columns = new_header
    replacement_dict = {'A': 1, 'G': 2, 'C': 3, 'T': 4, 'N': 0}
    df_transposed.replace(replacement_dict, inplace=True)
    df_numeric = df_transposed.apply(pd.to_numeric, errors='coerce')
    df_numeric.fillna(0, inplace=True)
    with open ('${pca}','rb') as f:
        pca = pickle.load(f)
    def decompose(new_data, expected_features):
        if isinstance(new_data, pd.DataFrame):
            new_data = new_data.values
        input_features = new_data.shape[1]
        if input_features > expected_features:
            new_data = new_data[:, :expected_features]
        elif input_features < expected_features:
            padding = np.zeros((new_data.shape[0], expected_features - input_features))
            new_data = np.hstack((new_data, padding))
        return new_data
    n_features = pca.n_features_in_
    processed_df = decompose(df_numeric,n_features)
    df_pca = pca.transform(processed_df)
    df_pca_df = pd.DataFrame(df_pca)
    df_pca_df.to_csv('${snp}.csv',index=False)
    "
    """
}

process predict {
    publishDir params.outdir, mode: 'copy'
    input:
    path new_data
    path model
	path labels
    output:
    path '*.txt'
    script:
    """
    python3 -c "
	import pandas as pd
	import pickle
	from sklearn.svm import SVC
	import numpy as np
	import warnings
	warnings.filterwarnings('ignore')
	new_data = pd.read_csv('${new_data}')
	with open('${model}','rb') as f:
		ml_model = pickle.load(f)
	input_data = new_data.iloc[:]
	pred = ml_model.predict(input_data)
	labels_list = []
	with open('${labels}','r') as f:
		for i in f:
			label = i.strip()
			labels_list.append(label)
	r = 'RESISTANT'
	s = 'SUSCEPTIBLE'
	pred_list = pred[0]
	with open('${new_data}_output.txt','w') as f:
		f.write('The predicted trait(s) for ${new_data} is:\\n')
		for i in range(0,len(pred_list)):
			out_trait = r + '\\n' if pred_list[i] == '0' else s + '\\n'
			antibiotic = labels_list[i]
			out_line = f'{antibiotic} {out_trait}'
			f.write(out_line)
    "
    """
}

process kmer {
    publishDir params.outdir, mode: 'copy'
    input:
    path fastq_file
    output:
    path '*.csv'
    script:
    """
    python3 -c "
    from Bio import SeqIO
    import gzip
    from collections import Counter
    import pandas as pd

    fastq_file_path = '${fastq_file}'
    if fastq_file_path.endswith('.gz'):
        handle = gzip.open(fastq_file_path, 'rt')
    else:
        handle = open(fastq_file_path, 'r')

    fastq_file = SeqIO.parse(handle,'fastq')

    k = int('${params.k}')

    kmer_counts = Counter()
    
    for record in fastq_file:
        sequence = str(record.seq)
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmer_counts[kmer] +=1

    df_kmer = pd.DataFrame(kmer_counts.items(), columns=['k-mer', 'Frequency'])
    df_kmer = df_kmer.transpose()
    df_kmer.to_csv(f'{fastq_file_path}_csv.csv',index=False, header=False)
    "
    """
}

process assemble {
    publishDir params.outdir, mode: 'copy'
    input:
    tuple path(read1), path(read2)
    output:
    path "${read1}_contigs.fasta"
    script:
    """
    spades -1 $read1 -2 $read2 -o ./
    mv contigs.fasta ${read1}_contigs.fasta
    """
}

process assemble_single_end {
	publishDir params.outdir, mode: 'copy'
    input:
    path reads
    output:
    path "${reads}_contigs.fasta"
    script:
    """
    spades -s $reads -o ./
    mv contigs.fasta ${reads}_contigs.fasta
    """
}

process kmer_fasta {
    publishDir params.outdir, mode: 'copy'
    input:
    path fasta_file
    output:
    path '*.csv'
    script:
    """
    python3 -c "
    from Bio import SeqIO
    import gzip
    from collections import Counter
    import pandas as pd

    fasta_file_path = '${fasta_file}'
    if fasta_file_path.endswith('.gz'):
        handle = gzip.open(fasta_file_path, 'rt')
    else:
        handle = open(fasta_file_path, 'r')

    fasta_file = SeqIO.parse(handle,'fasta')

    k = int('${params.k}')

    kmer_counts = Counter()
    
    for record in fasta_file:
        sequence = str(record.seq)
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            kmer_counts[kmer] +=1

    df_kmer = pd.DataFrame(kmer_counts.items(), columns=['k-mer', 'Frequency'])
    df_kmer = df_kmer.transpose()
    df_kmer.to_csv(f'{fasta_file_path}_csv.csv',index=False, header=False)
    "
    """
}

process genomeAnnotation {
    publishDir params.outdir, mode: 'copy'
	input:
	path fasta
	output:
	path "*.gff"
	script:
	"""
	prokka --outdir ./ --prefix $fasta $fasta --force
	sed -i '/##FASTA/,\$d; /^#/d' ${fasta}.gff
	"""
}

process mge_association {
    publishDir params.outdir, mode: 'copy'
    input:
    path mge
    path gff
    output:
    path "${mge}_mge_association.csv"
    script:
    """
    python3 -c "
    import pandas as pd
    mges = pd.read_csv('${mge}', comment='#')
    genes = pd.read_csv('${gff}', comment='#', sep='\\t', header=None,names=['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'])
    associations = []
    mges['start'] = mges['start'].astype(int)
    mges['end'] = mges['end'].astype(int)
    genes['start'] = genes['start'].astype(int)
    genes['end'] = genes['end'].astype(int)
    for _, mge in mges.iterrows():
        for _, gene in genes.iterrows():
            if gene['seqid'] == mge['contig']:  # Check if they are on the same contig
                if abs(gene['start'] - mge['start']) <= 31000 or abs(gene['end'] - mge['end']) <= 31000:
                    associations.append({
                        'MGE_id': mge['name'],
                            'MGE_type': mge['type'],
                        'Gene': gene['attributes'],
                        'MGE_START': mge['start'],
                        'MGE_END': mge['end'],
                        'GENE_START': gene['start'],
                        'GENE_END': gene['end'],
                        'Distance': min([abs(gene['start'] - mge['start']), abs(gene['end'] - mge['end'])])
                    })
    associations_df = pd.DataFrame(associations)
    associations_df.to_csv('${mge}_mge_association.csv', index=False)
    "
    """
}

process mgefind {
    publishDir params.outdir, mode: 'copy'
    input:
    path fasta
    output:
    path "${fasta}_mge.csv"
    script:
    """
    mefinder find -c $fasta ${fasta}_mge
    """
}

process argFind {
    publishDir params.outdir, mode: 'copy'
    input:
    path fasta
    output:
    path "*_results_tab.txt"
    script:
    """
    python3 -m resfinder -ifa $fasta -s \"${params.species}\" -acq -o ./
    """
}

process argAssociation {
    publishDir params.outdir, mode: 'copy'
    input:
    path arg
    path mge
    output:
    path "${mge}_arg_association.csv"
    script:
    """
    python3 -c "
    import pandas as pd
    args = pd.read_csv('${arg}', sep='\\t')
    mges = pd.read_csv('${mge}', comment='#')
    associations = []
    args = args[['Resistance gene', 'Position in contig']]
    args[['start', 'end']] = args['Position in contig'].str.split('\\.\\.', expand=True, n=1)
    args['start'] = args['start'].astype(int)
    args['end'] = args['end'].astype(int)
    args.drop(columns=['Position in contig'], inplace=True)
    mges['start'] = mges['start'].astype(int)
    mges['end'] = mges['end'].astype(int)
    for _, arg in args.iterrows():
        for _, mge in mges.iterrows():
            if abs(arg['start'] - mge['start']) <= 31000 or abs(arg['end'] - mge['end']) <= 31000:
                associations.append({
                    'ARG': arg['Resistance gene'],
                    'MGE': mge['name'],
                    'MGE_type': mge['type'],
                    'MGE_START': mge['start'],
                    'MGE_END': mge['end'],
                    'ARG_START': arg['start'],
                    'ARG_END': arg['end'],
                    'Distance': min([abs(arg['start'] - mge['start']), abs(arg['end'] - mge['end'])])
                })
    associations_df = pd.DataFrame(associations)
    associations_df.to_csv('${mge}_arg_association.csv', index=False)
    "
    """
}

workflow arg_mge_association {
    println "Workflow mode: To identify ARGs and MGEs associated with genes and ARGs"
    if (params.read1 != "" && params.read2 != "") {
            def paired_reads_channel = tuple(params.read1, params.read2)
        	trimming(paired_reads_channel)
    }
    else {
        paired_reads_channel = Channel.fromFilePairs("${params.reads}")
        paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] }
        trimming(paired_reads_channel)
    }
    assemble(trimming.out)
    mgefind(assemble.out)
    genomeAnnotation(assemble.out)
    argFind(assemble.out)
    argAssociation(argFind.out, mgefind.out)
    mge_association(mgefind.out, genomeAnnotation.out)
}

workflow arg_mge_association_single_end {
    println "Workflow mode: To identify ARGs and associated MGEs (Single End reads)"
    input_channel = Channel.from(params.read1.split(/\s+/))
    trimming_single_end(input_channel)
    assemble_single_end(trimming_single_end.out)
    mgefind(assemble_single_end.out)
    genomeAnnotation(assemble_single_end.out)
    argFind(assemble_single_end.out)
    argAssociation(argFind.out, mgefind.out)
    mge_association(mgefind.out, genomeAnnotation.out)
}

workflow arg {
    println "Workflow mode: To identify ARGs and associated MGEs"
    if (params.read1 != "" && params.read2 != "") {
            def paired_reads_channel = tuple(params.read1, params.read2)
        	trimming(paired_reads_channel)
    }
    else {
        paired_reads_channel = Channel.fromFilePairs("${params.reads}")
        paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] }
        trimming(paired_reads_channel)
    }
    assemble(trimming.out)
    mgefind(assemble.out)
    genomeAnnotation(assemble.out)
    argFind(assemble.out)
}

workflow mge{
    println "Workflow mode: To generate MGEs"
    if (params.read1 != "" && params.read2 != "") {
            def paired_reads_channel = tuple(params.read1, params.read2)
        	trimming(paired_reads_channel)
    }
    else {
        paired_reads_channel = Channel.fromFilePairs("${params.reads}")
        paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] }
        trimming(paired_reads_channel)
    }
    assemble(trimming.out)
    mgefind(assemble.out)
    genomeAnnotation(assemble.out)
    mge_association(mgefind.out, genomeAnnotation.out)
}

workflow mge_single_end{
    println "Workflow mode: To generate MGEs (Single End reads)"
    input_channel = Channel.from(params.read1.split(/\s+/))
    trimming_single_end(input_channel)
    assemble_single_end(trimming_single_end.out)
    genomeAnnotation(assemble_single_end.out)
    mgefind(assemble_single_end.out)
    mge_association(mgefind.out, genomeAnnotation.out)
}

workflow genome_annotation_single_end {
    println "Workflow mode: To generate genome annotation (Single End reads)"
	input_channel = Channel.from(params.read1.split(/\s+/))
	trimming_single_end(input_channel)
	assemble_single_end(trimming_single_end.out)
	genomeAnnotation(assemble_single_end.out)
}

workflow genome_annotation_paired_end {
    println "Workflow mode: To generate genome annotation (Paired end reads)"
    if (params.read1 != "" && params.read2 != "") {
            def paired_reads_channel = tuple(params.read1, params.read2)
        	trimming(paired_reads_channel)
    }
    else {
        paired_reads_channel = Channel.fromFilePairs("${params.reads}")
        paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] }
        trimming(paired_reads_channel)
    }
    assemble(trimming.out)
    genomeAnnotation(assemble.out)
}

workflow kmer_single_end {
    println "Workflow mode: To generate k-mer counts (Single End reads)"
    input_channel = Channel.from(params.read1.split(/\s+/))
    trimming_single_end(input_channel)
    kmer(trimming_single_end.out)
}

workflow kmer_paired_end {
    println "Workflow mode: To generate k-mer counts (Paired end reads)"
    if (params.read1 != "" && params.read2 != "") {
            def paired_reads_channel = tuple(params.read1, params.read2)
        	trimming(paired_reads_channel)
    }
    else {
        paired_reads_channel = Channel.fromFilePairs("${params.reads}")
        paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] }
        trimming(paired_reads_channel)
    }
    assemble(trimming.out)
    kmer_fasta(assemble.out)
}

workflow assembly_single_end {
    println "Workflow mode: To assemble single end reads"
    input_channel = Channel.from(params.read1.split(/\s+/))
    trimming_single_end(input_channel)
    assemble(trimming_single_end.out)
}

workflow assembly {
    println "Workflow mode: To assemble paired end reads"
    if (params.read1 != "" && params.read2 != "") {
            def paired_reads_channel = tuple(params.read1, params.read2)
        	trimming(paired_reads_channel)
    }
    else {
        paired_reads_channel = Channel.fromFilePairs("${params.reads}")
        paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] }
        trimming(paired_reads_channel)
    }
    //paired_reads_channel = paired_reads_channel.map { pair ->
    //(params.read1 != "" && params.read2 != "") ? [params.read1, params.read2] : [pair[1][0], pair[1][1]]}
    assemble(trimming.out)
}

workflow annotation_singleend{
	println "Workflow mode: To annotate the variants"
	println "Warning! Only one reads file was included. It will be assumed to be a single end file"
    indexReference(params.ref)
    input_channel = Channel.from(params.read1.split(/\s+/)).view()
    trimming_single_end(input_channel)
    alignReads_single_end(indexReference.out, trimming_single_end.out, params.ref)
    samToBam(alignReads_single_end.out)
    sorting(samToBam.out)
    variantCalling(sorting.out, params.ref)
    mutationAnnotation(variantCalling.out, params.bed)
    SNP(mutationAnnotation.out)
    SNPFiltering(SNP.out)
    SNPOutput(SNPFiltering.out)
    posRefAlt(SNPOutput.out)
    encode(posRefAlt.out, params.pca)
    predict(encode.out, params.model, params.labels)
}
	
workflow no_annotation_singleend{
	println "Workflow mode: To not annotate the variants (Default)"
	println "Warning! Only one reads file was included. It will be assumed to be a single end file"
    indexReference(params.ref)
    input_channel = Channel.from(params.read1.split(/\s+/)).view()
    trimming_single_end(input_channel)
    alignReads_single_end(indexReference.out, trimming_single_end.out,params.ref)
    samToBam(alignReads_single_end.out)
    sorting(samToBam.out)
    variantCalling(sorting.out,params.ref)
    SNP(variantCalling.out)
    SNPFiltering(SNP.out)
    SNPOutput(SNPFiltering.out)
    posRefAlt(SNPOutput.out)
    encode(posRefAlt.out, params.pca)
	predict(encode.out, params.model, params.labels)
	}

workflow no_annotation{
	println "Workflow mode: To not annotate the variants and bulk processing"
	println "Using wildcard rule: ${params.reads}"
    indexReference(params.ref)
    paired_reads_channel = Channel.fromFilePairs("${params.reads}")
    //paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] } //def reads_tuple = tuple(params.read1, params.read2)
    paired_reads_channel = paired_reads_channel.map { pair ->
    (params.read1 != "" && params.read2 != "") ? [params.read1, params.read2] : [pair[1][0], pair[1][1]]}
    trimming(paired_reads_channel)
    alignReads(indexReference.out, trimming.out,params.ref)
    samToBam(alignReads.out)
    sorting(samToBam.out)
    variantCalling(sorting.out,params.ref)
    SNP(variantCalling.out)
    SNPFiltering(SNP.out)
    SNPOutput(SNPFiltering.out)
    posRefAlt(SNPOutput.out)
    encode(posRefAlt.out, params.pca)
	predict(encode.out, params.model, params.labels)
	}

workflow annotation{
	println "Workflow mode: To annotate the variants and bulk processing"
    indexReference(params.ref)
    paired_reads_channel = Channel.fromFilePairs("${params.reads}")
    paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] }
    trimming(paired_reads_channel)
    alignReads(indexReference.out, trimming.out,params.ref)
    samToBam(alignReads.out)
    sorting(samToBam.out)
    variantCalling(sorting.out,params.ref)
    mutationAnnotation(variantCalling.out, params.bed)
    SNP(mutationAnnotation.out)
    SNPFiltering(SNP.out)
    SNPOutput(SNPFiltering.out)
    posRefAlt(SNPOutput.out)
    encode(posRefAlt.out, params.pca)
	predict(encode.out, params.model, params.labels)
	}

workflow process_all {
    println "Workflow mode: To process all"
    if (params.read1 != "" && params.read2 != "") {
            def paired_reads_channel = tuple(params.read1, params.read2)
        	trimming(paired_reads_channel)
    }
    else {
        paired_reads_channel = Channel.fromFilePairs("${params.reads}")
        paired_reads_channel = paired_reads_channel.map { pair -> [pair[1][0], pair[1][1]] }
        trimming(paired_reads_channel)
    }
    alignReads(indexReference.out, trimming.out,params.ref)
    samToBam(alignReads.out)
    sorting(samToBam.out)
    variantCalling(sorting.out,params.ref)
    mutationAnnotation(variantCalling.out, params.bed)
    SNP(mutationAnnotation.out)
    SNPFiltering(SNP.out)
    SNPOutput(SNPFiltering.out)
    posRefAlt(SNPOutput.out)
    encode(posRefAlt.out, params.pca)
    predict(encode.out, params.model, params.labels)
    assemble(trimming.out)
    kmer(assemble.out)
    genomeAnnotation(assemble.out)
    mgefind(assemble.out)
    mge_association(mgefind.out, genomeAnnotation.out)
}

workflow process_all_single_end {
    println "Workflow mode: To process all (Single End reads)"
    input_channel = Channel.from(params.read1.split(/\s+/))
    trimming_single_end(input_channel)
    alignReads_single_end(indexReference.out, trimming_single_end.out,params.ref)
    samToBam(alignReads_single_end.out)
    sorting(samToBam.out)
    variantCalling(sorting.out,params.ref)
    mutationAnnotation(variantCalling.out, params.bed)
    SNP(mutationAnnotation.out)
    SNPFiltering(SNP.out)
    SNPOutput(SNPFiltering.out)
    posRefAlt(SNPOutput.out)
    encode(posRefAlt.out, params.pca)
    predict(encode.out, params.model, params.labels)
    assemble_single_end(trimming_single_end.out)
    kmer(trimming_single_end.out)
    genomeAnnotation(assemble_single_end.out)
    mgefind(assemble_single_end.out)
    mge_association(mgefind.out, genomeAnnotation.out)
}

workflow {
    def isPairedEnd = params.read2 != "" || params.reads != ""
    def isSingleEnd = params.read2 == "" && params.reads == ""
    def isAnnotation = params.annotation == "yes"
    def isSnpMode = params.mode == "snp"
    def isKmerMode = params.mode == "kmer"
    def isAssembleMode = params.mode == "assemble"
    def isGenomeAnnotationMode = params.mode == "genome_annotation"
    def isMgeMode = params.mode == "mge"
    def isSupermode = params.mode == "super"

    if (isSnpMode) {
        if (isSingleEnd) {
            if (isAnnotation) {
                annotation_singleend()
            } else {
                no_annotation_singleend()
            }
        } else {
            if (isAnnotation) {
                if (params.reads != "") {
                    annotation()
                }
            } else {
                if (params.reads != "") {
                    no_annotation()
                }
            }
        }
    } else if (isKmerMode) {
        if (isPairedEnd) {
            kmer_paired_end()
        } else {
            kmer_single_end()
        }
    } else if (isAssembleMode) {
        if (isSingleEnd) {
            assembly_single_end()
        } else {
            assembly()
        }
    } else if (isGenomeAnnotationMode) {
        if (isSingleEnd) {
            genome_annotation_single_end()
        } else {
            genome_annotation_paired_end()
        }
    } else if (isMgeMode) {
        if (isSingleEnd) {
            mge_single_end()
        } else {
            mge()
        }
    }
    else if (isSupermode) {
        if (isSingleEnd) {
            process_all_single_end()
        } else {
            process_all()
        }
    }
    else if (params.mode == "arg_mge" || params.mode == "mge_arg") {
        if (isSingleEnd) {
            arg_mge_association_single_end()
        } else {
            arg_mge_association()
        }
    } else if (params.mode == "arg") {
        arg()
    }
    else {
        println "Invalid mode"
    }
}
