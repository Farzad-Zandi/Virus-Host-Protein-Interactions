## Farzad Zandi, 2025.
# Feature Extraction.
rm(list=ls())

library(dplyr)
library(protr)
library(ftrCOOL)
library(data.table)

# Load train data.
# Data 1.
data1 <- read.table('D:/Research/Virus Host PPIs/train_and_test/train_set_group1', header = TRUE)

# Data 2.
data2 <- read.table('D:/Research/Virus Host PPIs/train_and_test/train_set_group2', header = TRUE)

# Data 3.
data3 <- read.table('D:/Research/Virus Host PPIs/train_and_test/train_set_group3', header = TRUE)

# Merge train data.
dataTrain = rbind(data1, data2, data3)
dataTrain <- dataTrain %>% distinct(Human_ID, Virus_ID, .keep_all = TRUE)

# Remove extra variables.
rm(data1, data2, data3)

# Load Test data.
# Data 1.
data1 <- read.table('D:/Research/Virus Host PPIs/train_and_test/independent_test_group1', header = TRUE)

# Data 2.
data2 <- read.table('D:/Research/Virus Host PPIs/train_and_test/independent_test_group2', header = TRUE)

# Data 3.
data3 <- read.table('D:/Research/Virus Host PPIs/train_and_test/independent_test_group3', header = TRUE)

# Merge test data.
dataTest = rbind(data1, data2, data3)
dataTest <- dataTest %>% distinct(Human_ID, Virus_ID, .keep_all = TRUE)

# Remove extra variables.
rm(data1, data2, data3)  

# Merge train and test data.
allData <- rbind(dataTrain, dataTest)
allData <- allData %>% distinct(Human_ID, Virus_ID, .keep_all = TRUE)

# Extract features.
CT = c()
DDE = c()
PseAAC = c()
DC = c()

for (i in 1:575239)
{
  seq <- allData$Virus_Seq[i]
  out <- extractCTriad(seq) # Extract Conjoint Triad Features.
  CT <- rbind(CT, out)
  
  out = PSEAAC(seq, lambda = 11) # Extract Pseudo Amino Acid Composition Features.
  PseAAC <- rbind(PseAAC, out)
  
  out <- DDE(seq) # Extract Dipeptide Deviation from Expected Mean Features.
  DDE <- rbind(DDE, out)
  
  out <- extractDC(seq) # Extract Dipeptide Composition Features.
  DC <- rbind(DC, out)
}

virus_CT <- as.data.frame(CT)
virus_DC <- as.data.frame(DC)
virus_DDE <- as.data.frame(DDE)
virus_PseAAC <- as.data.frame(PseAAC)

# Save new data.
write.csv(virus_CT, '/virus_CT.csv')
...
