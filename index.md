Building and Implementing an LDA Topic Model: UK Annual Reports
================
Furqan Shah
2024-10-23

# Introduction

In this tutorial, we will apply Latent Dirichlet Allocation (LDA) topic
modeling on data extracted from 23 companies’ annual reports (available
in the repository for this tutorial). The dataset is self-annotated,
with narratives classified into the following categories for each
company’s annual report: **Financial**, **Human**, **Intellectual**,
**Natural**, **Social & Relationship**, **Manufactured**, and an
**Unclassified** category.

This tutorial will guide you through performing Latent Dirichlet
Allocation (LDA) topic modeling in R using the `topicmodels` package. We
will cover:

- Loading and preprocessing text data
- Creating a Document-Term Matrix (DTM)
- Performing LDA topic modeling
- Visualizing the results

# Loading Libraries

We begin by loading the necessary R libraries for the tasks listed
above.

``` r
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
```

## Step 1 - Load Data

The data consists of 23 separate `.tsv` files, each representing a
different company. Each file contains two columns: `Text` (which holds
sentences or paragraphs) and `Theme` (the category assigned to the
text). The categories in `Theme` column include: **Financial**,
**Human**, **Intellectual**, **Natural**, **Social & Relationship**, and
**Unclassified**, as defined below:

- **Financial**: *Information about the company’s financial performance,
  including revenue, expenses, profits, and financial health.*
- **Human**: *Relates to employees, their well-being, skills, and
  workforce management.*
- **Intellectual**: *Covers intangible assets like patents, trademarks,
  and the company’s innovation capacity.*
- **Natural**: *Focuses on environmental sustainability, resource usage,
  and the company’s ecological impact.*
- **Social & Relationship**: *Refers to the company’s interactions with
  stakeholders, including communities, customers, and regulators.*
- **Manufactured**: *Involves physical assets such as infrastructure,
  equipment, and operational capacity.*
- **Unclassified**: *Includes disclosures that do not fit into the other
  categories.*

``` r
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
```

    ## Rows: 9,711
    ## Columns: 3
    ## $ text   <chr> "Notably, our production performance in all our key businesses …
    ## $ theme  <chr> "Unclassified", "Unclassified", "Human", "Human", "Intellectual…
    ## $ source <chr> "2014_angloamerican.tsv", "2014_angloamerican.tsv", "2014_anglo…

``` r
data %>% count(source) # number of source files (should be 23)
```

    ##                      source   n
    ## 1    2014_angloamerican.tsv 444
    ## 2       2014_baesystems.tsv 482
    ## 3              2014_bat.tsv 498
    ## 4      2014_britishland.tsv 330
    ## 5          2014_btgroup.tsv 511
    ## 6         2014_cocacola.tsv 446
    ## 7            2014_crest.tsv 307
    ## 8           2014_diageo.tsv 574
    ## 9        2014_fresnillo.tsv 693
    ## 10             2014_gog.tsv 392
    ## 11        2014_halfords.tsv 438
    ## 12       2014_hammerson.tsv 422
    ## 13          2014_howden.tsv 208
    ## 14            2014_hsbc.tsv 420
    ## 15             2014_ihg.tsv 485
    ## 16          2014_lloyds.tsv 297
    ## 17   2014_marksnspencer.tsv 302
    ## 18      2014_mediclinic.tsv 592
    ## 19           2014_mondi.tsv 318
    ## 20            2014_sage.tsv 504
    ## 21        2014_unilever.tsv 375
    ## 22 2014_unitedutilities.tsv 368
    ## 23        2014_vodafone.tsv 305

``` r
data %>% count(theme) # number of categories assigned to text (should be 7)  
```

    ##                 theme    n
    ## 1           Financial 1851
    ## 2               Human  704
    ## 3        Intellectual 1590
    ## 4        Manufactured  501
    ## 5             Natural  575
    ## 6 Social_Relationship 1929
    ## 7        Unclassified 2561

## Step 2 - Data Preprocessing

In this step, we preprocess the text data by removing special characters
and numbers, and tokenize it into individual words, resulting in one
word per row. We also exclude the “Unclassified” category from our
dataset, as it is not relevant to this project.

- Text Cleaning: Remove special characters and numbers from the text.
- Tokenization: Split the text into individual words (tokens).
- Stop Words Removal: Remove common English stop words that do not carry
  significant meaning (such as “the”, “for”, “with”, “from” etc.)
- Filtering: Remove words with fewer than four characters and entries
  labeled as “Unclassified”.
- Counting: Count the frequency of each word within each theme, and
  filter for words that occur more than 3 times.

``` r
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
```

    ## Rows: 8,368
    ## Columns: 3
    ## $ theme <chr> "Financial", "Financial", "Financial", "Financial", "Financial",…
    ## $ word  <chr> "2012underlying", "2013us", "2014us", "5total", "9total", "aamea…
    ## $ n     <int> 5, 4, 6, 4, 5, 5, 30, 6, 4, 8, 11, 20, 6, 18, 10, 29, 23, 7, 23,…

## Step 3 - Create a Document-Term-Matrix (DTM)

The function `cast_dtm` coverts the text data into a
Document-Term-Matrix.

``` r
# Cast dtm
dtm <- data2 %>%
  cast_dtm(document = theme,
           term = word,
           value = n,
           weighting = tm::weightTf)

# Display the DTM
dtm
```

    ## <<DocumentTermMatrix (documents: 6, terms: 3566)>>
    ## Non-/sparse entries: 8368/13028
    ## Sparsity           : 61%
    ## Maximal term length: 19
    ## Weighting          : term frequency (tf)

## Step 4 - Run an LDA Model with 6 Topics

In this step, we run an LDA model specifying six topics, corresponding
to the six narrative reporting themes in the dataset. Our aim is to
determine whether the LDA model can identify topics that align with
these six narrative reporting themes.

``` r
# Set the number of topics
k_topics <- 6

# Perform LDA
lda <- LDA(dtm, k = k_topics, control = list(seed = 10))
```

The two primary outputs of LDA are commonly referred to as “beta” and
“gamma” matrices. These matrices are essential for interpreting the
results of the LDA model, as they provide insights into the
relationships between topics, words, and documents.

## Step 5 - Visualizing Word-Topic Probabilities (Beta)

In LDA topic modeling, the beta matrix represents the probability of
each word belonging to each topic, essentially showing how strongly each
word is associated with each topic. It helps identify the most
significant words that define each topic by listing the words with the
highest probabilities within that topic.

``` r
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
```

![](index_files/figure-gfm/LDA%20beta-1.png)<!-- -->

In the plot above, we see the words with the highest probabilities for
each of the six topics identified by the LDA model. For example, Topic 4
includes top words such as *water*, *emissions*, *energy*, *carbon*,
*environmental*, and *waste.* This suggests that Topic 4 corresponds to
environmental reporting, aligning with the Natural reporting category in
our data. Similarly, Topic 6 features words like *employees*, *people*,
*training*, *business*, *development*, *management*, and *skills*,
indicating a focus on human capital reporting. We can further confirm
these associations in the next step when we examine the gamma matrix,
which shows the probability of each topic occurring in each document
(i.e., each narrative reporting category in our data).

## Step 6 - Visualizing Topic-Document Probabilities (Gamma)

The gamma matrix represents the probability of each topic being present
in each document, indicating how strongly each topic is associated with
each document. It helps understand the thematic composition of each
document by showing the distribution of topics across the documents in
the corpus.

``` r
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
```

![](index_files/figure-gfm/LDA%20gamma-1.png)<!-- -->

By analyzing the gamma plot, we observe that Topics 4 and 6, which we
previously identified in the beta plot, have the highest probabilities
of occurring in the Natural and Human reporting categories,
respectively. This alignment makes sense and indicates that the LDA
model has been implemented appropriately, effectively capturing the
underlying themes within our data.

# Further Example - LDA Topic Modeling on Natural Capital Reporting

In this section, we shift our focus to a subset of the data where the
theme is “Natural”. By isolating this category, we aim to explore the
specific topics discussed within natural capital reporting. Instead of
analyzing all reporting themes collectively, this targeted approach
allows us to delve deeper into the distinct topics present exclusively
within the natural capital narratives.

``` r
# Create DTM for 'Natural' theme
dtm_natural <- data2 %>%
  filter(theme == "Natural") %>%
  cast_dtm(document = theme,
           term = word,
           value = n,
           weighting = tm::weightTf)

# Display the DTM
dtm_natural
```

    ## <<DocumentTermMatrix (documents: 1, terms: 813)>>
    ## Non-/sparse entries: 813/0
    ## Sparsity           : 0%
    ## Maximal term length: 15
    ## Weighting          : term frequency (tf)

``` r
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
```

![](index_files/figure-gfm/LDA%20natural-1.png)<!-- -->

By analyzing the LDA results for the natural capital category, we
observe three topics in the plot:

- **Topic 1** features words like emissions, carbon, and waste, which
  indicate a focus on environmental pollutants and their management.
  Additionally, terms such as business, management, and operations imply
  an organizational approach to handling these issues. Therefore, this
  topic appears to capture **Environmental Emissions and Operational
  Management**.

- **Topic 2** includes words like energy, efficiency, and consumption,
  pointing toward discussions on energy usage and efficiency measures.
  The presence of water and environmental suggests a broader
  consideration of resource management. Words like data, emissions, and
  management indicate a focus on tracking, reporting, and managing
  environmental impact. Thus, we can name this topic **Energy Efficiency
  and Resource Consumption**.

- **Topic 3** contains words such as climate, change, carbon, and
  reduction, which clearly point toward climate change mitigation
  efforts. The emphasis on reduction underscores efforts to lower
  negative environmental impacts. Consequently, we might name this topic
  **Climate Change Mitigation and Sustainability Initiatives**.

# Conclusion

In this tutorial, we’ve covered how to perform LDA topic modeling in R
using the `topicmodels` package. We loaded and preprocessed text data,
created a Document-Term Matrix, ran LDA to extract topics, and
visualized the results. By adjusting the number of topics and focusing
on specific themes, you can gain deeper insights into the underlying
structure of your text data.
