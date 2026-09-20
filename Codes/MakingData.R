# ============================================================
# Farzad Zandi, 2026
# Dataset Construction
# ============================================================

rm(list = ls())
data <- read.csv('/V_H_ppi_data.csv', header = TRUE)

# Identify positive and negative interaction pairs
posIdx <- which(data$allData.Label == 1)
negIdx <- which(data$allData.Label == -1)

pos_data <- data[posIdx, ]

# ============================================================
# Construct CT-based datasets
# ============================================================

neg_data <- data[negIdx[1:22653], ]
data1 <- rbind(pos_data, neg_data)
write.csv(data1,'/DATA1.csv', row.names = FALSE)

neg_data <- data[negIdx[22654:...], ]
data2 <- rbind(pos_data, neg_data)
write.csv(data2, '/DATA2.csv', row.names = FALSE)

# Continue constructing Data3–Data25 using the same procedure.

# ============================================================
# Construct Encoded Sequence-Feature Datasets
# ============================================================

rm(list = ls())
library(jsonlite)
data <- read.csv('/V_H_ppi_data.csv', header = TRUE)

posIdx <- which(data$allData.Label == 1)
negIdx <- which(data$allData.Label == -1)

encoded_data <- readRDS("/All_Merged_encoded_values.rds")

pos_data <- encoded_data[posIdx]

# Data 1
neg_data <- encoded_data[negIdx[1:22653]]

data1 <- c(  pos_data,
  neg_data
)

write_json(data1,"/DATA1_Encoded.json", pretty = TRUE)

# Data 2
neg_data <- encoded_data[negIdx[22654:...]]

data2 <- c(pos_data, neg_data)

write_json(data2, "/DATA2_Encoded.json", pretty = TRUE)

# Continue constructing Data3–Data25 using the same procedure.

# ============================================================
# Construct GO-Based Datasets from One-Hot-Encoded GO Terms
# ============================================================

rm(list = ls())
library(dplyr)

data <- read.csv('V_H_ppi_data.csv', header = TRUE)

# Load one-hot-encoded GO representations
virus_GO_Encoded <- read.csv('/virus_go_encoded.csv')
virus_GO_Encoded <- virus_GO_Encoded[-1]

human_GO_Encoded <- read.csv('/human_go_encoded.csv')
human_GO_Encoded <- human_GO_Encoded[-1]

# ------------------------------------------------------------
# Retain interaction pairs with available GO annotations
# ------------------------------------------------------------

filtered_data <- data[
  data$allData.Virus_ID %in%
    virus_GO_Encoded$Protein_ID,
]

filtered_data <- filtered_data[
  filtered_data$allData.Human_ID %in%
    human_GO_Encoded$Protein_ID,
]

filtered_data <- filtered_data[, 2:4]

rm(data)

colnames(filtered_data) <- c(
  "label",
  "humanID",
  "virusID"
)

colnames(human_GO_Encoded)[
  colnames(human_GO_Encoded) == "Protein_ID"
] <- "humanID"

colnames(virus_GO_Encoded)[
  colnames(virus_GO_Encoded) == "Protein_ID"
] <- "virusID"

# ------------------------------------------------------------
# Separate positive and negative interaction pairs
# ------------------------------------------------------------

posIdx <- which(filtered_data$label == 1)
negIdx <- which(filtered_data$label == -1)

pos_data <- filtered_data[posIdx, ]
neg_data <- filtered_data[negIdx, ]

rm(posIdx, negIdx, filtered_data)

# ------------------------------------------------------------
# Construct Data 1
# ------------------------------------------------------------

pos_GO <- pos_data %>%
  left_join(
    human_GO_Encoded,
    by = "humanID"
  ) %>%
  left_join(
    virus_GO_Encoded,
    by = "virusID"
  )

neg_IDX <- neg_data[1:19891, ]

neg_GO <- neg_IDX %>%
  left_join(
    human_GO_Encoded,
    by = "humanID"
  ) %>%
  left_join(
    virus_GO_Encoded,
    by = "virusID"
  )

data1 <- rbind(pos_GO, neg_GO)
write.csv(data1, '/DATA1_GO.csv', row.names = FALSE)
rm(data1, neg_GO, neg_IDX)

# ------------------------------------------------------------
# Construct Data 2
# ------------------------------------------------------------

neg_IDX <- neg_data[19892:..., ]

neg_GO <- neg_IDX %>%
  left_join(
    human_GO_Encoded,
    by = "humanID"
  ) %>%
  left_join(
    virus_GO_Encoded,
    by = "virusID"
  )

data2 <- rbind(pos_GO, neg_GO)
write.csv(data2, '/DATA2_GO.csv', row.names = FALSE)
rm(data2, neg_GO, neg_IDX)

# Continue constructing Data3–Data24 using the same procedure.
