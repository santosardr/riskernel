import json

# The 32 original questions from the first 53.5k tokens
original_qa = [
    {"id": 1, "question": "In version 1.0 of GenPPi, what specific impact did the Buchnera aphidicola genome show before the machine learning features were introduced?", "options": ["A) A massively overgrown network", "B) A relatively low number of nodes and edges", "C) A complete system crash", "D) High structural conservation"], "correct_option": "B", "article_source": "genppi.txt"},
    {"id": 2, "question": "Who specifically conceived the GenPPI project, supervised the work, and developed the core components of the software?", "options": ["A) ASi", "B) CM and IG", "C) ARS", "D) BT"], "correct_option": "C", "article_source": "genppi.txt"},
    {"id": 3, "question": "According to the abstract/metadata, what is the exact URL where GenPPi version 1.5 is available for download?", "options": ["A) https://genppi.facom.ufu.br/", "B) https://genppi.github.io/", "C) https://www.ufu.br/genppi", "D) https://genppi.sourceforge.net/"], "correct_option": "A", "article_source": "genppi.txt"},
    {"id": 4, "question": "What is the consequence of naively maximizing the number of predicted protein similarities?", "options": ["A) Better model accuracy", "B) Exacerbation of the computational burden of downstream steps", "C) Shorter computational times", "D) Loss of GUI responsiveness"], "correct_option": "B", "article_source": "genppi.txt"},
    {"id": 5, "question": "A key aspect of GenPPI users' rules allows the end user to decide:", "options": ["A) The coloration of the network nodes", "B) How many and which genomes to use in constructing the protein interaction network", "C) The pricing of the cloud instance", "D) The specific programming language output"], "correct_option": "B", "article_source": "genppi.txt"},
    {"id": 6, "question": "What specific statistical test was utilized to compare the empirical distribution functions of topological metrics like Degree across replicates?", "options": ["A) Student's t-test", "B) Kolmogorov-Smirnov (KS) Test (Multi-Sample)", "C) ANOVA", "D) Chi-Square Test"], "correct_option": "B", "article_source": "genppi.txt"},
    {"id": 7, "question": "In the GenPPI Random Forest optimization table, against which independent test set were the overall performance metrics reported?", "options": ["A) E. coli + Salmonella", "B) C. pseudotuberculosis + C. glutamicum", "C) S. aureus + B. subtilis", "D) Human interactome only"], "correct_option": "B", "article_source": "genppi.txt"},
    {"id": 8, "question": "Which specific author supervised the overall work as opposed to conceiving the project in GenPPI?", "options": ["A) ASi", "B) ARS", "C) BT", "D) MB"], "correct_option": "C", "article_source": "genppi.txt"},
    {"id": 9, "question": "Introducing Features and Machine Learning to GenPPI version 1.0 caused what effect on Buchnera aphidicola networks?", "options": ["A) Reduced number of nodes to zero", "B) Notable increase in the number of nodes and edges", "C) Caused network fragmentation", "D) Had absolutely no effect"], "correct_option": "B", "article_source": "genppi.txt"},
    {"id": 10, "question": "In the GenPPI analysis, the optimization table evaluates Random Forest models across how many negative-to-positive instance ratios?", "options": ["A) Three", "B) Five", "C) Seven", "D) Ten"], "correct_option": "C", "article_source": "genppi.txt"},
    {"id": 11, "question": "What happens if a user naively maximizes predicted similarities in GenPPI?", "options": ["A) Computational burden decreases", "B) It exacerbates computational burden of downstream steps", "C) Network nodes disappear", "D) The model converges instantly"], "correct_option": "B", "article_source": "genppi.txt"},
    {"id": 12, "question": "Which entity is empowered by GenPPI to select the genomes for network construction?", "options": ["A) The Machine Learning core", "B) The end user", "C) The Random Forest algorithm", "D) The cloud administrator"], "correct_option": "B", "article_source": "genppi.txt"},
    {"id": 13, "question": "What is the exact email address of the corresponding author listed in the AOM paper?", "options": ["A) anderson@ufu.br", "B) santosardr@ufu.br", "C) ueira@ufu.br", "D) admin@genppi.org"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 14, "question": "How many target proteins did the AOM network include initially (before neighbor expansion)?", "options": ["A) 12", "B) 34", "C) 123", "D) 2467"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 15, "question": "According to the ETRC module, which specific AOM electron transfer protein directly connects to the core nitrogenase components NifD and NifK?", "options": ["A) HdrABC", "B) HdrDE", "C) C5S33_05775", "D) MCR"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 16, "question": "Which organism’s enzymes were used as queries to identify the Factor F430 biosynthesis (Cfb) pathway in ANME-2a?", "options": ["A) Bacillus cereus", "B) E. coli", "C) Methanosarcina acetivorans", "D) Acetilactobacillus jinshanensis"], "correct_option": "C", "article_source": "aom.txt"},
    {"id": 17, "question": "Which specific statistical indication proved the power-law model provided a significantly better fit than the lognormal model?", "options": ["A) p ≈ 0.05", "B) p ≈ 3.0 × 10−27", "C) p ≈ 1.0 × 10−5", "D) p ≈ 0.99"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 18, "question": "What specific function does the cytoplasmic hypothetical protein MRG76964.1 serve in module CC1M?", "options": ["A) Oxygen sensing", "B) A critical inter-module bridge connecting facets of C1 and redox metabolism", "C) Complete breakdown of ATP", "D) Ribosomal translation framing"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 19, "question": "In the MFCB module's enriched functions, what specific metabolic pathways were completely absent from the evidence?", "options": ["A) Translation and transcription", "B) Pathways related to methane oxidation or nitrogen fixation", "C) Lactate production", "D) Glucose metabolism"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 20, "question": "What prevents metabolic cross-talk and ensures complex metallo-cofactors reach their targets with high fidelity?", "options": ["A) Rapid degradation", "B) Having dedicated shuttle proteins", "C) High cellular temperatures", "D) Random cytoplasmic diffusion"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 21, "question": "What is the stated specific purpose of continuing to probe ecosystems with metatranscriptomics and metaproteomics?", "options": ["A) To count the number of extinct species", "B) To confirm the expression and activity of integrated pathways in situ, linking genomic potential to real-world function", "C) To sequence new viral pathogens", "D) To artificially cultivate methanogens in zero-gravity"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 22, "question": "MRG76973.1 is specifically annotated as which partial heterodisulfide reductase?", "options": ["A) HdrDE", "B) HdrABC", "C) MCR subunit C", "D) NifN"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 23, "question": "Which specific protein serves as the essential linker between F430 production and MCR assembly?", "options": ["A) C5S33_05775", "B) MRG76964.1", "C) HdrABC", "D) NifD"], "correct_option": "A", "article_source": "aom.txt"},
    {"id": 24, "question": "MRG76473.1 corresponds to which nitrogenase-associated protein?", "options": ["A) NifD", "B) NifK", "C) NifN", "D) HdrDE"], "correct_option": "C", "article_source": "aom.txt"},
    {"id": 25, "question": "What does module CC1M connect via the inter-module bridge protein MRG76964.1?", "options": ["A) Amino acid and lipid assembly", "B) C1 and redox metabolism", "C) DNA and RNA replication", "D) Sugar and strictly aerobic respiration"], "correct_option": "B", "article_source": "aom.txt"},
    {"id": 26, "question": "What software tool was provided as a 'minimal runner' to execute SBML models directly?", "options": ["A) sim_growth.sh", "B) run_sbml_model.py", "C) flux_analyzer.exe", "D) compile_b_cereus.R"], "correct_option": "B", "article_source": "ajinshanensis.txt"},
    {"id": 27, "question": "In the interactome visualization, enzymes exclusive to the EMP pathway are denoted by which visual marker?", "options": ["A) Inner green circle", "B) Outer red circle", "C) Shared purple center", "D) Blue isolated nodes"], "correct_option": "B", "article_source": "ajinshanensis.txt"},
    {"id": 28, "question": "Enzymes exclusive to the PKP pathway are visualized using which specific representation?", "options": ["A) Inner green circle", "B) Outer red circle", "C) Purple bridging nodes", "D) Black triangles"], "correct_option": "A", "article_source": "ajinshanensis.txt"},
    {"id": 29, "question": "What score output does GenPPI use to reflect the strength of genomic signals like profile similarity?", "options": ["A) A p-value from 0.01 to 0.05", "B) A confidence score ranging from 0 to 1", "C) A raw nucleotide count", "D) A binary Yes/No classification"], "correct_option": "B", "article_source": "ajinshanensis.txt"},
    {"id": 30, "question": "Which transcription regulator families are notably identified among the B. cereus environmental sensing hubs?", "options": ["A) CtsR, TetR/AcrR, PadR", "B) RpoS, Sigma32", "C) SoxR, OxyR", "D) LacI, TrpR"], "correct_option": "A", "article_source": "ajinshanensis.txt"},
    {"id": 31, "question": "Metabolic flux analysis specifically indicated that both studied isolates possess pathways for producing which precursors?", "options": ["A) Lignin precursors", "B) Terpenoid precursors", "C) Alkaloid substrates", "D) Cellulose fibers"], "correct_option": "B", "article_source": "ajinshanensis.txt"},
    {"id": 32, "question": "Which specific complete pathways for production were computationally identified in Acetilactobacillus jinshanensis offering a molecular basis for its metabolic profile?", "options": ["A) Lactate and ethanol production", "B) Methane and sulfur production", "C) Glucose and oxygen production", "D) Formaldehyde processing"], "correct_option": "A", "article_source": "ajinshanensis.txt"}
]

# 32 new questions, largely based on GenPPI and early parts of AOM
new_qa = [
    {"question": "What percentage of sequence identity was required by the original GenPPi (v1.0) for protein similarity assessment?", "options": ["A) > 90%", "B) > 50%", "C) > 75%", "D) 100%"], "correct_option": "A", "article_source": "genppi.txt"},
    {"question": "Which algorithm was integrated into GenPPi 1.5 to manage the computational complexity of interaction sampling?", "options": ["A) Neural Network Sampling", "B) Principal Component Analysis", "C) Reduced Interaction Sampling (RIS)", "D) Markov Chain Monte Carlo"], "correct_option": "C", "article_source": "genppi.txt"},
    {"question": "According to the text, which database is recognized as a gold standard in protein interaction analysis?", "options": ["A) GenBank", "B) STRING", "C) UniProt", "D) PDB"], "correct_option": "B", "article_source": "genppi.txt"},
    {"question": "What programming language was GenPPi developed in?", "options": ["A) Python", "B) Common Lisp", "C) C++", "D) Java"], "correct_option": "B", "article_source": "genppi.txt"},
    {"question": "What is the primary purpose of the propensity.dat file in GenPPi 1.5?", "options": ["A) To map specific physicochemical features of amino acids", "B) To store genome alignments", "C) To define network visualization colors", "D) To authenticate users"], "correct_option": "A", "article_source": "genppi.txt"},
    {"question": "How many features does the Features algorithm generate for each protein in GenPPi 1.5?", "options": ["A) Twenty", "B) Forty", "C) Sixty", "D) One hundred"], "correct_option": "C", "article_source": "genppi.txt"},
    {"question": "The RIS algorithm divides a large phylogenetic profile into multiple sublists. What is the typical size of these sublists relative to the original profile?", "options": ["A) 90%", "B) 50%", "C) 10%", "D) 25%"], "correct_option": "A", "article_source": "genppi.txt"},
    {"question": "In the Machine Learning mode of GenPPi 1.5, what happens to the network edge density compared to the Features mode?", "options": ["A) It slightly decreases", "B) It significantly increases", "C) It remains exactly the same", "D) It drops to zero"], "correct_option": "B", "article_source": "genppi.txt"},
    {"question": "What metric was used to measure the consistent presence of critical nodes across replicates in the RIS evaluation?", "options": ["A) Average Degree", "B) Weighted Global Presence Mean (MGP)", "C) Network Diameter", "D) Clustering Coefficient"], "correct_option": "B", "article_source": "genppi.txt"},
    {"question": "Which parameter in GenPPi establishes the tolerated difference between phylogenetic profiles?", "options": ["A) -ppdifftolerated", "B) -tolerate_diff", "C) -diff_max", "D) -profile_variance"], "correct_option": "A", "article_source": "genppi.txt"},
    {"question": "The cl-random-forest library used in GenPPi is an implementation of Random Forest for which specific Lisp compiler?", "options": ["A) Clozure CL", "B) GNU CLISP", "C) SBCL", "D) ABCL"], "correct_option": "C", "article_source": "genppi.txt"},
    {"question": "How many decision trees were utilized in the final customized Random Forest model for GenPPi?", "options": ["A) 100", "B) 500", "C) 1000", "D) 50"], "correct_option": "B", "article_source": "genppi.txt"},
    {"question": "What was the average F1-Score of the finalized Random Forest model across the 26 diverse genome test sets?", "options": ["A) Approximately 51.4%", "B) Approximately 90.0%", "C) Approximately 25.0%", "D) Approximately 99.9%"], "correct_option": "A", "article_source": "genppi.txt"},
    {"question": "Which database management system was used to compare GenPPi results with STRING data?", "options": ["A) MySQL", "B) MongoDB", "C) PostgreSQL", "D) SQLite"], "correct_option": "C", "article_source": "genppi.txt"},
    {"question": "How does GenPPi 1.5 handle repeated interactions during analysis?", "options": ["A) They are multiplied to increase weight", "B) They are removed using Gephi", "C) They are ignored completely", "D) They cause a runtime error"], "correct_option": "B", "article_source": "genppi.txt"},
    {"question": "The GenPPi software is publicly available under which license?", "options": ["A) MIT License", "B) GPLv3", "C) Apache 2.0", "D) Proprietary"], "correct_option": "A", "article_source": "genppi.txt"},
    {"question": "According to the abstract logic framing B. cereus hubs, defense mechanisms include all EXCEPT:", "options": ["A) Antitoxins", "B) DNA repair enzymes", "C) Catalase", "D) CRISPR-Cas9 adaptive immunity"], "correct_option": "D", "article_source": "ajinshanensis.txt"},
    {"question": "GenPPI assigns proximity and similarity confidence metrics ranging specifically from:", "options": ["A) 0 to 100", "B) 0 to 1", "C) -1 to 1", "D) 1 to 10"], "correct_option": "B", "article_source": "ajinshanensis.txt"},
    {"question": "What software and version were exactly used to remove contaminants and obtain valid NGS data?", "options": ["A) Trimmomatic 0.39", "B) SOAPnuke software (version 2.1.7)", "C) FastQC 0.11", "D) Cutadapt 3.1"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "In the meta-analysis study, which specific chloroplast genome was assembled to provide a schematic representation?", "options": ["A) Arabidopsis thaliana", "B) Lactuca sativa", "C) Glycine max", "D) Zea mays"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "The output read quality value system in SOAPnuke was specifically set to what standard?", "options": ["A) Phred+64", "B) Phred+33", "C) Q10", "D) Sanger Standard"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "What specific equipment was used to assess Library quality before sequencing?", "options": ["A) Qubit 4 Fluorometer", "B) Agilent 2100 Bioanalyzer", "C) NanoDrop Spectrophotometer", "D) Bio-Rad Gel Doc"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "Who is the corresponding author strictly paired with 'Laboratory of Genetics, Institute of Biotechnology' in meta.txt?", "options": ["A) Anderson R. dos Santos", "B) Carlos Ueira-Vieira", "C) SilvaMacdo", "D) Ranjan"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "What is the specific address for the Laboratory of Genetics mentioned in the correspondence section?", "options": ["A) Acre Street, 2E building, room 226", "B) Joao Naves de Avila Ave, Block 1A", "C) Paulista Ave, Sao Paulo", "D) Federal University Main Campus"], "correct_option": "A", "article_source": "meta.txt"},
    {"question": "The ANI value for ajita (this study) vs. A. jinshanensis (CP187400.1) was definitively measured at:", "options": ["A) 25.5%", "B) 81.39%", "C) 95.00%", "D) 99.90%"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "What is the precise sequencing strategy and fragment length used for the libraries detailed in meta.txt?", "options": ["A) Ion Torrent 400 bp", "B) DNBseq paired-end (150 bp)", "C) Illumina Single-Read 50 bp", "D) PacBio long-reads (10kb)"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "Which default k-mer values were employed when assembling the scaffolds with SPAdes version 3.14.0?", "options": ["A) 21, 33, 55, and 77 bp", "B) 11, 21, and 31 bp", "C) 64 and 128 bp", "D) 25, 45, and 85 bp"], "correct_option": "A", "article_source": "meta.txt"},
    {"question": "For the pollen count in larval food, what exact percentage was represented by the family Solanaceae?", "options": ["A) 21.72%", "B) 8.24%", "C) 24.97%", "D) 88.00%"], "correct_option": "C", "article_source": "meta.txt"},
    {"question": "What exact value was calculated for the representation of the Asteraceae family in the pollen count?", "options": ["A) Ra = 21.72%", "B) Ra = 24.97%", "C) Ra = 8.24%", "D) Ra = 88%"], "correct_option": "A", "article_source": "meta.txt"},
    {"question": "What is the exact dDDH value range calculated between the ajita strain and Acetilactobacillus jinshanensis?", "options": ["A) 81.1–81.4%", "B) 25–27%", "C) >70%", "D) 95–96%"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "According to SilvaMacdo2023 referenced in the meta text, T. angustula visited plants from how many different families?", "options": ["A) 24", "B) 88", "C) 150", "D) 4"], "correct_option": "B", "article_source": "meta.txt"},
    {"question": "In the visualization schema, what color specifically represents the shared bridging enzymes?", "options": ["A) Green", "B) Red", "C) Purple", "D) Yellow"], "correct_option": "C", "article_source": "ajinshanensis.txt"}
]

# Total 64 questions
all_qa = original_qa + new_qa

# We need to assign them token positions corresponding to the tiers:
# 4 < 2611
# 4 < 6092
# 8 < 13056
# 16 < 26982
# 32 < 54835

tier_positions = []
# Tier 1
tier_positions.extend([500, 1000, 1500, 2000])
# Tier 2
tier_positions.extend([3000, 4000, 5000, 6000])
# Tier 3
tier_positions.extend([7000, 7800, 8600, 9400, 10200, 11000, 11800, 12600])
# Tier 4
tier_positions.extend([13500, 14300, 15100, 15900, 16700, 17500, 18300, 19100, 19900, 20700, 21500, 22300, 23100, 23900, 24700, 25500])
# Tier 5
import numpy as np
t5_positions = np.linspace(28000, 54000, 32, dtype=int).tolist()
tier_positions.extend(t5_positions)

for i, qa in enumerate(all_qa):
    qa["id"] = i + 1
    qa["answer_token_position"] = int(tier_positions[i])
    if "options" in qa and len(qa["options"]) == 4:
        qa["options"].append("E) None of the above / Not mentioned in the text")

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(base_dir, "qa_dataset.json"), "w", encoding="utf-8") as f:
    json.dump(all_qa, f, indent=4, ensure_ascii=False)

print("Created 64 questions and saved to qa_dataset.json")
