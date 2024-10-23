# List of required packages
packages <- c("dplyr", "tidyr", "stringr", "tidytext", "topicmodels", "ggplot2")

# Install packages that are not yet installed
installed_packages <- rownames(installed.packages())

for (pkg in packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}


# Set the path to your data files
path <- "F:/Github Projects/LDA-in-R/data files" #Note: Replace the path variable with the path to your own data directory

# Load and combine data from all TSV files in the directory
data <- list.files(path = path, pattern = "\\.tsv$", full.names = TRUE) %>%
  lapply(function(file) read.delim(file) %>% mutate(source_file = basename(file))) %>%
  bind_rows() %>%
  rename(text = Text,
         theme = Theme,
         source = source_file) %>%
  mutate(across(everything(), ~na_if(str_trim(.), ""))) %>%
  drop_na()

# Preview the data structure
glimpse(data)

data %>% count(source) # number of source files (should be 23)
data %>% count(theme) # number of categories assigned to text (should be 7) 


# Clean the text data and remove unwanted characters
data2 <- data %>%
  mutate(text = gsub("[?âèåäã'’ˆ'½æ_ï&îˆ/.$,%'-']", " ", text)) %>% # remove special characters
  mutate(text = gsub("([0-9])([a-z])", "\\1 \\2", text)) %>%        # create space between numbers and words
  mutate(text = gsub("([a-z])([0-9])", "\\1 \\2", text)) %>%        # create space between words and numbers
  mutate(text = gsub("[^[:alnum:]]", " ", text)) %>%                # replace each non-alphanumeric character with a single space
  unnest_tokens(word, text) %>%
  filter(!str_detect(word, "^[0-9]*$")) %>%
  anti_join(stop_words) %>%
  mutate(characters = nchar(word)) %>%
  filter(characters > 3, !theme == "Unclassified") %>%
  count(theme, word) %>%
  filter(n > 3)

# Preview the processed data
glimpse(data2)


# Cast dtm
dtm <- data2 %>%
  cast_dtm(document = theme,
           term = word,
           value = n,
           weighting = tm::weightTf)

# Display the DTM
dtm

# Set the number of topics
k_topics <- 6

# Perform LDA
lda <- LDA(dtm, k = k_topics, control = list(seed = 10))


# Extract the beta matrix
lda_beta <- tidy(lda, matrix = "beta") %>%
  arrange(desc(beta))

# Get the top 10 terms for each topic
top_terms_topic <- lda_beta %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Plot the top terms for each topic
topics_beta_plot <- top_terms_topic %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered()

# Display the plot
topics_beta_plot


# Extract the gamma matrix
lda_gamma <- tidy(lda, matrix = "gamma")

# Plot the gamma values
topics_gamma_plot <- lda_gamma %>%
  mutate(document = reorder(document, gamma * topic)) %>%
  ggplot(aes(factor(topic), gamma)) +
  geom_boxplot() +
  facet_wrap(~ document)

# Display the plot
topics_gamma_plot

# Create DTM for 'Natural' theme
dtm_natural <- data2 %>%
  filter(theme == "Natural") %>%
  cast_dtm(document = theme,
           term = word,
           value = n,
           weighting = tm::weightTf)

# Display the DTM
dtm_natural

# Set the number of topics
k_natural_topics <- 3 # you may try with different numbers of topics as it is context specific

# Perform LDA
lda_natural <- LDA(dtm_natural, k = k_natural_topics, control = list(seed = 10))

# Extract the beta matrix
lda_beta_natural <- tidy(lda_natural, matrix = "beta") %>%
  arrange(desc(beta))

# Get the top 10 terms for each topic
top_terms_topic_natural <- lda_beta_natural %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

# Plot the top terms for each topic
topics_beta_plot_natural <- top_terms_topic_natural %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered()

# Display the plot
topics_beta_plot_natural