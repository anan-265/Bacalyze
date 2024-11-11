#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.ref = ''
params.read1 = ''
params.read2 = ''
params.qual = '15'
newline = '\\n'
params.pca = '/home/ananthu/Desktop/Rfastq/pca.pkl'
params.model = '/home/ananthu/Desktop/Rfastq/cip_svm.pkl'
params.outdir = ''

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

process trimming {
	input:
		Integer qual
		path raw1
		path raw2
	
	output:
		tuple path("${raw1}_trimmed.fastq"), path("${raw2}_trimmed.fastq")
	
	script:
	"""
	echo "Trimming raw data. Inputs: ${raw1} and ${raw2} Quality: ${qual}"
	fastp -i $raw1 -I $raw2 -o ${raw1}_trimmed.fastq -O ${raw2}_trimmed.fastq
	"""

}

process alignReads {
    input:
    path ref_index
	tuple path(forward), path(reverse)
    path reference
	path fastq
	
    output:
    path "${fastq}*"

    script:
        """
	echo "Aligning Reads. Reference: ${reference}"
	bwa mem -p $reference $forward $reverse > ${fastq}.sam
        """
}

process samToBam {
	input:
		path sam
		path fastq
	
	output:
		path "${fastq}*"
	
	script:
		"""
		samtools view -Sb $sam > ${fastq}.bam
		"""
	
}

process sorting {
	input:
		path bam
		path fastq
	
	output:
		path "${fastq}*"
	
	script:
		"""
		samtools sort $bam -o ${fastq}_sorted.bam
		"""

}

process variantCalling {
	input:
		path sorted
		path fastq
		path reference
	
	output:
		path "${fastq}*"
	
	script:
		"""
		bcftools mpileup -f $reference $sorted | bcftools call -vm -Oz > ${fastq}.vcf.gz
		"""
}

process SNP {
	input:
		path variant
		path fastq
	
	output:
		path "${fastq}*"
	
	script:
		"""
		vcftools --gzvcf $variant --minDP 4 --max-missing 0.2 --minQ 30 --recode --recode-INFO-all --out ${fastq}.filter
		"""
}

process SNPFiltering {
	
	input:
		path snp
		path fastq
	output:
		path "${fastq}*"
	
	script:
		"""
		vcftools --gzvcf $snp --remove-indels --recode --recode-INFO-all --out ${fastq}.filter.snps
		echo "Task completed successfully. The output file is ${fastq}.filter.snps.recode.vcf"
		"""
}

process SNPOutput {

	input:
		path snps
		path fastq
	
	output:
		path "${fastq}*"
	
	script:
		"""
		bcftools query -f '%CHROM %POS %REF %ALT [%TGT]\n' $snps -o ${fastq}.filter.snps.extract.txt
		"""

}

process posRefAlt {

	
	input:
		path snpout
		path fastq
	
	output:
		path "${fastq}*"
	
	script:
		"""
		cut -d " " -f 2,4 $snpout > ${fastq}.snp
		"""
}

process encode {

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
pd.set_option('future.no_silent_downcasting', True)
df = pd.read_csv('${snp}', sep=' ', header=None)
df_transposed = df.transpose()
new_header = df_transposed.iloc[0] 
df_transposed = df_transposed[1:] 
df_transposed.columns = new_header 
replacement_dict = {'A': 1, 'G': 2, 'C': 3, 'T': 4, 'N': 0}
df_transposed.replace(replacement_dict, inplace=True)
#df_transposed.to_csv('sample.csv',index=False)
df_numeric = df_transposed.apply(pd.to_numeric, errors='coerce')
df_numeric.fillna(0, inplace=True)
#df_numeric.to_csv('df_numeric.csv',index=False)
with open ('pca.pkl','rb') as f:
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
df_pca_df.to_csv('input.csv',index=False)
	"
	"""
}
process predict{
	publishDir params.outdir, mode: 'copy'
	input:
		path new_data
		path model
		path fastq
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
new_data = pd.read_csv('input.csv')
with open('cip_svm.pkl','rb') as f:
	ml_model = pickle.load(f)
input_data = new_data.iloc[:,1:]
pred = ml_model.predict(input_data)
print(str(pred[0]))
if pred[0] == 0:
	result_final = 'The predicted trait for the given sequence against CIPROFLOXACIN is SUSCEPTIBLE\\n'
elif pred[0] == 1:
	result_final = 'The predicted trait for the given sequence against CIPROFLOXACIN is RESISTANCE\\n'
with open('${fastq}output.txt','w') as f:
	f.write(result_final)
"
	"""
}



workflow {
    indexReference(params.ref)
	trimming(params.read1,params.read2)
    alignReads(indexReference.out, trimming.out,params.ref,params.read1)
    samToBam(alignReads.out,params.read1)
    sorting(samToBam.out,params.read1)
    variantCalling(sorting.out,params.read1,params.ref)
    SNP(variantCalling.out,params.read1)
    SNPFiltering(SNP.out,params.read1)
    SNPOutput(SNPFiltering.out,params.read1)
    posRefAlt(SNPOutput.out,params.read1)
    encode(posRefAlt.out, params.pca)
	predict(encode.out,params.model,params.read1)
	}
