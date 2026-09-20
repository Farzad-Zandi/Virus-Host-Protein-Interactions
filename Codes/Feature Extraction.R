# ============================================================
# Farzad Zandi, 2026
# Protein Sequence Feature Extraction
# ============================================================

rm(list = ls())

# ------------------------------------------------------------
# Load required packages
# ------------------------------------------------------------
library(dplyr)
library(protr)
library(ftrCOOL)
library(data.table)

# ============================================================
# Load Training Data
# ============================================================

# Training data: Group 1
data1 <- read.table('/train_set_group1', header = TRUE)

# Training data: Group 2
data2 <- read.table('/train_set_group2',  header = TRUE)

# Training data: Group 3
data3 <- read.table('/train_set_group3',  header = TRUE)

# ------------------------------------------------------------
# Merge training datasets and remove duplicate interactions
# ------------------------------------------------------------
dataTrain <- rbind(data1, data2, data3)

dataTrain <- dataTrain %>%
  distinct(
    Human_ID,
    Virus_ID,
    .keep_all = TRUE
  )

# Release temporary objects
rm(data1, data2, data3)

# ============================================================
# Load Independent Test Data
# ============================================================

# Independent test data: Group 1
data1 <- read.table(/independent_test_group1', header = TRUE)

# Independent test data: Group 2
data2 <- read.table(/independent_test_group2', header = TRUE)

# Independent test data: Group 3
data3 <- read.table(/independent_test_group3', header = TRUE)

# ------------------------------------------------------------
# Merge independent test datasets and remove duplicates
# ------------------------------------------------------------
dataTest <- rbind(data1, data2, data3)

dataTest <- dataTest %>%
  distinct(
    Human_ID,
    Virus_ID,
    .keep_all = TRUE
  )

# Release temporary objects
rm(data1, data2, data3)

# ============================================================
# Combine Training and Test Data
# ============================================================
allData <- rbind(dataTrain, dataTest)

allData <- allData %>%
  distinct(
    Human_ID,
    Virus_ID,
    .keep_all = TRUE
  )

# ============================================================
# Extract Protein Sequence Features
# ============================================================

# Initialize feature matrices
CT <- c()
DDE <- c()
PseAAC <- c()
DC <- c()

# ------------------------------------------------------------
# Extract sequence-based features from viral proteins
# The same feature extraction procedure is applied to host protein
# ------------------------------------------------------------

for (i in 1:575239) {

  seq <- allData$Virus_Seq[i]

  # Conjoint Triad (CT) features
  out <- extractCTriad(seq)
  CT <- rbind(
    CT,
    out
  )

  # Pseudo-Amino Acid Composition (PseAAC) features
  out <- PSEAAC(
    seq,
    lambda = 11
  )

  PseAAC <- rbind(
    PseAAC,
    out
  )

  # Dipeptide Deviation from Expected Mean (DDE) features
  out <- DDE(seq)
  DDE <- rbind(
    DDE,
    out
  )

  # Dipeptide Composition (DC) features
  out <- extractDC(seq)
  DC <- rbind(
    DC,
    out
  )
}

# ============================================================
# Convert Extracted Features to Data Frames
# ============================================================
virus_CT <- as.data.frame(CT)
virus_DC <- as.data.frame(DC)
virus_DDE <- as.data.frame(DDE)
virus_PseAAC <- as.data.frame(PseAAC)


# ============================================================
# Save Extracted Features
# ============================================================

write.csv(
  virus_CT,
  '/virus_CT.csv',
  row.names = FALSE
)

write.csv(
  virus_DC,
  '/virus_DC.csv',
  row.names = FALSE
)

write.csv(
  virus_DDE,
  '/virus_DDE.csv',
  row.names = FALSE
)

write.csv(
  virus_PseAAC,
  '/virus_PseAAC.csv',
  row.names = FALSE
)

# ============================================================
# Feature Extraction Completed
# ============================================================

cat("\n")
cat("============================================================\n")
cat("Protein sequence feature extraction completed successfully.\n")
cat("Generated representations: CT, DC, DDE, and PseAAC.\n")
cat("============================================================\n")
