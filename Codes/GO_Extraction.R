## Farzad Zandi, 2025.
# Fetching Go Terms from UniProt.

library(httr)
library(tibble)
library(tidyr)
library(dplyr)

rm(list=ls())

data <- read.csv('/all_virus_host_PPI.csv', header = TRUE)
human <- unique(data$Human_ID)
#virus <- unique(data$Virus_ID)

# Function to get GO terms from UniProt
get_go_terms <- function(protein_id)
{
  protein_id <- sub("-.*", "", protein_id)
  url <- paste0("https://www.uniprot.org/uniprot/", protein_id, ".txt")
  response <- GET(url)
  content <- content(response, "text")
  # Extract the GO terms using regex
  go_terms <- regmatches(content, gregexpr("GO:[0-9]+", content, perl=TRUE))[[1]]
  return(go_terms)
}

human_GO <- tibble(Protein_ID = character(), GO_Terms = character())

for (pid in human) 
{
  terms <- get_go_terms(pid)
  human_GO <- add_row(human_GO, Protein_ID = pid, GO_Terms = paste(terms, collapse = "; "))
  print(i)
  i<-i+1
}