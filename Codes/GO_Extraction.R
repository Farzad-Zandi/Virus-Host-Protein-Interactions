# ============================================================
# Farzad Zandi, 2026
# Retrieval of Gene Ontology (GO) Terms from UniProt
# ============================================================

library(httr)
library(tibble)
library(dplyr)

rm(list = ls())

# Load virus-host interaction data
data <- read.csv('/V_H_PPI.csv', header = TRUE)


# Extract unique host protein identifiers
human <- unique(data$Human_ID)
virus <- unique(data$virus_ID)
# ------------------------------------------------------------
# Retrieve GO terms from UniProt
# ------------------------------------------------------------
get_go_terms <- function(protein_id) {
  protein_id <- sub("-.*", "", protein_id)
  url <- paste0(
    "https://www.uniprot.org/uniprot/",
    protein_id,
    ".txt"
  )
  response <- GET(url)
  content <- content(response, "text")

  # Extract GO identifiers
  go_terms <- regmatches(
    content,
    gregexpr(
      "GO:[0-9]+",
      content,
      perl = TRUE
    )
  )[[1]]

  return(go_terms)
}

# ------------------------------------------------------------
# Retrieve GO annotations for host proteins
# The same procedure is applied for viral proteins.
# ------------------------------------------------------------
human_GO <- tibble(Protein_ID = character(), GO_Terms = character())
i <- 1

for (pid in human) {
  terms <- get_go_terms(pid)
  human_GO <- add_row(
    human_GO,
    Protein_ID = pid,
    GO_Terms = paste(
      terms,
      collapse = "; "
    )
  )

  print(
    paste(
      "Processed:",
      i,
      "/",
      length(human)
    )
  )

  i <- i + 1
}

print("GO annotation retrieval completed.")
