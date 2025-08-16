## Farzad Zandi, 2025.
# Making dataset.

rm(list=ls())

data <- read.csv('/all_virus_host_ppi_data.csv', header = TRUE)
posIdx <- which(data$allData.Label==1)
negIdx <- which(data$allData.Label==-1)

pos_data <- data[posIdx,]

neg_data <- data[negIdx[1:22653],]
data1 <- rbind(pos_data, neg_data)
write.csv(data1, '/data1_CT.csv')

neg_data <- data[negIdx[22654:...],]
data2 <- rbind(pos_data, neg_data)
write.csv(data2, '/data2_CT.csv')

...

##########################
## Making Encoded Values Data 1-25.

rm(list=ls())
library(jsonlite)

data <- read.csv('/all_virus_host_ppi_data.csv', header = TRUE)
posIdx <- which(data$allData.Label==1)
negIdx <- which(data$allData.Label==-1)

encoded_data <- readRDS("/All_Merged_encoded_values.rds")

pos_data <- encoded_data[posIdx]

neg_data <- encoded_data[negIdx[1:22653]]
data1 <- c(pos_data, neg_data)
write_json(data1, "/data1_Encoded.json", pretty = TRUE)

neg_data <- data[negIdx[22654:...],]
data2 <- c(pos_data, neg_data)
write_json(data2, "/data2_Encoded.json", pretty = TRUE)

...

#######################################################
# Making data from One Hot Encoded Go Terms.
rm(list=ls())
library(dplyr)

data <- read.csv('/all_virus_host_ppi_data.csv', header = TRUE)

virus_GO_Encoded <- read.csv('/virus_go_encoded.csv')
virus_GO_Encoded <- virus_GO_Encoded[-1]
human_GO_Encoded <- read.csv('/human_go_encoded.csv')
human_GO_Encoded <- human_GO_Encoded[-1]

filtered_data <- data[(data$allData.Virus_ID %in% virus_GO_Encoded$Protein_ID), ]
filtered_data <- filtered_data[(filtered_data$allData.Human_ID %in% human_GO_Encoded$Protein_ID), ]
filtered_data <- filtered_data[,2:4]

rm(data)

colnames(filtered_data) = c("label", "humanID", "virusID")
colnames(human_GO_Encoded)[colnames(human_GO_Encoded)=="Protein_ID"] = "humanID"
colnames(virus_GO_Encoded)[colnames(virus_GO_Encoded)=="Protein_ID"] = "virusID"

posIdx <- which(filtered_data$label ==1)
negIdx <- which(filtered_data$label ==-1)

pos_data <- filtered_data[posIdx,]
neg_data <- filtered_data[negIdx,]

rm(posIdx, negIdx, filtered_data)

pos_GO <- pos_data %>%
  left_join(human_GO_Encoded, by = "humanID") %>%
  left_join(virus_GO_Encoded, by = "virusID")

neg_IDX <- neg_data[1:19891,]
neg_GO <- neg_IDX %>%
  left_join(human_GO_Encoded, by = "humanID") %>%
  left_join(virus_GO_Encoded, by = "virusID")

data1 <- rbind(pos_GO, neg_GO)
write.csv(data1 ,'/data1_GO.csv')
rm(data1, neg_GO, neg_IDX)

neg_IDX <- neg_data[19892:...,]
neg_GO <- neg_IDX %>%
  left_join(human_GO_Encoded, by = "humanID") %>%
  left_join(virus_GO_Encoded, by = "virusID")

data2 <- rbind(pos_GO, neg_GO)
write.csv(data2 ,'/data2_GO.csv')
rm(data2, neg_GO, neg_IDX)

...
