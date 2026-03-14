#!/bin/bash
# Common SNP Strategy for TIGAR

echo "🎯 COMMON SNP STRATEGY FOR TIGAR"
echo "================================"
echo ""

CLEAN_VCF="migraine/Fold_0/TrainVCF/chr1_clean.vcf.gz"
WEIGHT_FILE="/data/ascher02/uqmmune1/ANNOVAR/TIGAR/weights/Weights/Brain_Nucleus_accumbens_basal_ganglia/DPR_CHR1/CHR1_DPR_train_eQTLweights.txt.gz"

echo "1. FILTER FOR COMMON SNPs ONLY"
echo "==============================="

echo "Creating common SNP VCF (MAF >= 0.05)..."
bcftools view -q 0.05 "$CLEAN_VCF" -Oz -o migraine/Fold_0/TrainVCF/chr1_common.vcf.gz
tabix -p vcf migraine/Fold_0/TrainVCF/chr1_common.vcf.gz

if [ -f "migraine/Fold_0/TrainVCF/chr1_common.vcf.gz" ]; then
    COMMON_COUNT=$(bcftools view -H migraine/Fold_0/TrainVCF/chr1_common.vcf.gz | wc -l)
    echo "✅ Common SNPs VCF created: $COMMON_COUNT variants"
    
    echo "Position range of common SNPs:"
    MIN_COMMON=$(bcftools query -f '%POS\n' migraine/Fold_0/TrainVCF/chr1_common.vcf.gz | sort -n | head -1)
    MAX_COMMON=$(bcftools query -f '%POS\n' migraine/Fold_0/TrainVCF/chr1_common.vcf.gz | sort -n | tail -1)
    echo "  Min: $MIN_COMMON"
    echo "  Max: $MAX_COMMON"
else
    echo "❌ Failed to create common SNPs VCF"
    exit 1
fi
echo ""

echo "2. TEST SNP OVERLAP WITH COMMON SNPs"
echo "===================================="

# Test overlap with common SNPs only
COMMON_VCF_SNPS=$(mktemp)
WEIGHT_SNPS=$(mktemp)

echo "Extracting common SNPs from VCF..."
bcftools query -f '%CHROM:%POS:%REF:%ALT\n' migraine/Fold_0/TrainVCF/chr1_common.vcf.gz > "$COMMON_VCF_SNPS"

echo "Extracting SNPs from GTEx weights..."
zcat "$WEIGHT_FILE" | tail -n +2 | awk -F'\t' '{print $1":"$2":"$4":"$5}' > "$WEIGHT_SNPS"

COMMON_OVERLAP=$(comm -12 <(sort "$COMMON_VCF_SNPS") <(sort "$WEIGHT_SNPS") | wc -l)
echo "Overlap with common SNPs: $COMMON_OVERLAP"

if [ "$COMMON_OVERLAP" -gt 22 ]; then
    echo "✅ IMPROVED! $COMMON_OVERLAP overlapping common SNPs (was 22 total)"
    echo "Sample overlapping common SNPs:"
    comm -12 <(sort "$COMMON_VCF_SNPS") <(sort "$WEIGHT_SNPS") | head -5
else
    echo "⚠️  Still only $COMMON_OVERLAP overlapping common SNPs"
fi

# Clean up
rm -f "$COMMON_VCF_SNPS" "$WEIGHT_SNPS"
echo ""

echo "3. TRY TIGAR WITH COMMON SNPs"
echo "============================="

echo "Testing TIGAR with common SNPs and relaxed parameters..."

timeout 180s bash /data/ascher02/uqmmune1/ANNOVAR/TIGAR/TIGAR_GReX_Pred.sh \
--format GT \
--gene_anno /data/ascher02/uqmmune1/ANNOVAR/TIGAR/gene_anno.txt \
--test_sampleID migraine/Fold_0/TrainVCF/training_sampleID.txt \
--chr 1 \
--weight "$WEIGHT_FILE" \
--genofile migraine/Fold_0/TrainVCF/chr1_common.vcf.gz \
--genofile_type vcf \
--window 2000000 \
--missing_rate 0.5 \
--maf_diff 0.5 \
--thread 1 \
--out_dir migraine/Fold_0/TigarTrainExpression/Brain_common_test \
--TIGAR_dir /data/ascher02/uqmmune1/ANNOVAR/TIGAR

echo ""
echo "4. ALTERNATIVE: VERY RELAXED FILTERING"
echo "======================================"

echo "Creating VCF with very relaxed filtering (MAF >= 0.01)..."
bcftools view -q 0.01 "$CLEAN_VCF" -Oz -o migraine/Fold_0/TrainVCF/chr1_relaxed.vcf.gz
tabix -p vcf migraine/Fold_0/TrainVCF/chr1_relaxed.vcf.gz

echo "Testing TIGAR with relaxed filtering..."

timeout 180s bash /data/ascher02/uqmmune1/ANNOVAR/TIGAR/TIGAR_GReX_Pred.sh \
--format GT \
--gene_anno /data/ascher02/uqmmune1/ANNOVAR/TIGAR/gene_anno.txt \
--test_sampleID migraine/Fold_0/TrainVCF/training_sampleID.txt \
--chr 1 \
--weight "$WEIGHT_FILE" \
--genofile migraine/Fold_0/TrainVCF/chr1_relaxed.vcf.gz \
--genofile_type vcf \
--window 5000000 \
--missing_rate 0.8 \
--maf_diff 0.8 \
--thread 1 \
--out_dir migraine/Fold_0/TigarTrainExpression/Brain_relaxed_test \
--TIGAR_dir /data/ascher02/uqmmune1/ANNOVAR/TIGAR

echo ""
echo "5. CHECK GENE-SPECIFIC OVERLAP"
echo "=============================="

echo "Checking if overlapping SNPs are in gene regions..."

# Get the 22 overlapping SNPs
ALL_VCF_SNPS=$(mktemp)
ALL_WEIGHT_SNPS=$(mktemp)

bcftools query -f '%CHROM:%POS:%REF:%ALT\n' "$CLEAN_VCF" > "$ALL_VCF_SNPS"
zcat "$WEIGHT_FILE" | tail -n +2 | awk -F'\t' '{print $1":"$2":"$4":"$5}' > "$ALL_WEIGHT_SNPS"

OVERLAP_SNPS=$(mktemp)
comm -12 <(sort "$ALL_VCF_SNPS") <(sort "$ALL_WEIGHT_SNPS") > "$OVERLAP_SNPS"

echo "The 22 overlapping SNPs are:"
cat "$OVERLAP_SNPS"

echo ""
echo "Checking which genes these SNPs belong to..."
while IFS= read -r snp; do
    # Extract position
    pos=$(echo "$snp" | cut -d':' -f2)
    # Find genes that include this position
    zcat "$WEIGHT_FILE" | awk -F'\t' -v pos="$pos" '$2 == pos {print $6}' | sort -u
done < "$OVERLAP_SNPS" | sort -u | head -10 | while read gene; do
    echo "  Gene: $gene"
done

# Clean up
rm -f "$ALL_VCF_SNPS" "$ALL_WEIGHT_SNPS" "$OVERLAP_SNPS"
echo ""

echo "6. SUMMARY AND RECOMMENDATIONS"
echo "=============================="

echo "PROGRESS MADE:"
echo "✅ Found 22 overlapping SNPs (proves compatible reference builds)"
echo "✅ Same coordinate system (hg38 most likely)"
echo "✅ Clean VCF without coordinate issues"
echo ""

echo "REMAINING ISSUE:"
echo "❌ Need MORE overlapping SNPs for TIGAR to work"
echo "❌ Only 22 out of 48,971 SNPs overlap (0.045%)"
echo ""

echo "NEXT STRATEGIES:"
echo "🔧 1. Use broader SNP filtering (less stringent MAF)"
echo "🔧 2. Use larger window sizes around genes"
echo "🔧 3. Try different GTEx tissue types (may have different SNP sets)"
echo "🔧 4. Consider using PredictDB weights (better SNP coverage)"
echo "🔧 5. Use imputed data if available"
echo ""

echo "🎯 BEST IMMEDIATE SOLUTION:"
echo "Try PredictDB weights which are designed for broader compatibility"
echo "Download from: http://predictdb.org/post/2021/07/21/gtex-v8-models-on-eqtl-and-sqtl/"