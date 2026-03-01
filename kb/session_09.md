
Session 9: Synthetic Data Generation and RAG Evaluation with LangSmith
In the following notebook we'll explore a use-case for RAGAS' synthetic testset generation workflow, and use it to evaluate and iterate on a RAG pipeline with LangSmith!

Learning Objectives:

Understand Ragas' knowledge graph-based synthetic data generation workflow
Generate synthetic test sets with different query synthesizer types
Load synthetic data into LangSmith for evaluation
Evaluate a RAG chain using LangSmith evaluators
Iterate on RAG pipeline parameters and measure the impact
Table of Contents:
Breakout Room #1: Synthetic Data Generation with Ragas

Task 1: Dependencies and API Keys
Task 2: Data Preparation and Knowledge Graph Construction
Task 3: Generating Synthetic Test Data
Question #1 & Question #2
🏗️ Activity #1: Custom Query Distribution
Breakout Room #2: RAG Evaluation with LangSmith

Task 4: LangSmith Dataset Setup
Task 5: Building a Basic RAG Chain
Task 6: Evaluating with LangSmith
Task 7: Modifying the Pipeline and Re-Evaluating
Question #3 & Question #4
🏗️ Activity #2: Analyze Evaluation Results
🤝 Breakout Room #1
Synthetic Data Generation with Ragas
Task 1: Dependencies and API Keys
We'll need to install a number of API keys and dependencies, since we'll be leveraging a number of great technologies for this pipeline!

OpenAI's endpoints to handle the Synthetic Data Generation
OpenAI's Endpoints for our RAG pipeline and LangSmith evaluation
QDrant as our vectorstore
LangSmith for our evaluation coordinator!
Let's install and provide all the required information below!

Dependencies and API Keys:
NLTK Import
To prevent errors that may occur based on OS - we'll import NLTK and download the needed packages to ensure correct handling of data.

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
[nltk_data] Error loading punkt: <urlopen error [SSL:
[nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed:
[nltk_data]     unable to get local issuer certificate (_ssl.c:1028)>
[nltk_data] Error loading averaged_perceptron_tagger: <urlopen error
[nltk_data]     [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
[nltk_data]     failed: unable to get local issuer certificate
[nltk_data]     (_ssl.c:1028)>
False
import os
import getpass

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")
We'll also want to set a project name to make things easier for ourselves.

from uuid import uuid4

os.environ["LANGCHAIN_PROJECT"] = f"AIM - SDG - {uuid4().hex[0:8]}"
OpenAI's API Key!

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
Generating Synthetic Test Data
We wil be using Ragas to build out a set of synthetic test questions, references, and reference contexts. This is useful because it will allow us to find out how our system is performing.

NOTE: Ragas is best suited for finding directional changes in your LLM-based systems. The absolute scores aren't comparable in a vacuum.

Data Preparation
We'll prepare our data using two complementary guides — a Health & Wellness Guide covering exercise, nutrition, sleep, and stress management, and a Mental Health & Psychology Handbook covering mental health conditions, therapeutic approaches, resilience, and daily mental health practices. The topical overlap between documents helps RAGAS build rich cross-document relationships in the knowledge graph.

Next, let's load our data into a familiar LangChain format using the TextLoader.

from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
docs = loader.load()
print(f"Loaded {len(docs)} documents: {[d.metadata['source'] for d in docs]}")
Loaded 2 documents: ['data/MentalHealthGuide.txt', 'data/HealthWellnessGuide.txt']
Knowledge Graph Based Synthetic Generation
Ragas uses a knowledge graph based approach to create data. This is extremely useful as it allows us to create complex queries rather simply. The additional testset complexity allows us to evaluate larger problems more effectively, as systems tend to be very strong on simple evaluation tasks.

Let's start by defining our generator_llm (which will generate our questions, summaries, and more), and our generator_embeddings which will be useful in building our graph.

Unrolled SDG
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
Next, we're going to instantiate our Knowledge Graph.

This graph will contain N number of nodes that have M number of relationships. These nodes and relationships (AKA "edges") will define our knowledge graph and be used later to construct relevant questions and responses.

from ragas.testset.graph import KnowledgeGraph

kg = KnowledgeGraph()
kg
KnowledgeGraph(nodes: 0, relationships: 0)
The first step we're going to take is to simply insert each of our full documents into the graph. This will provide a base that we can apply transformations to.

from ragas.testset.graph import Node, NodeType

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )
kg
KnowledgeGraph(nodes: 2, relationships: 0)
Now, we'll apply the default transformations to our knowledge graph. This will take the nodes currently on the graph and transform them based on a set of default transformations.

These default transformations are dependent on the corpus length, in our case:

Producing Summaries -> produces summaries of the documents
Extracting Headlines -> finding the overall headline for the document
Theme Extractor -> extracts broad themes about the documents
It then uses cosine-similarity and heuristics between the embeddings of the above transformations to construct relationships between the nodes.

from ragas.testset.transforms import default_transforms, apply_transforms

transformer_llm = generator_llm
embedding_model = generator_embeddings

default_transforms = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
apply_transforms(kg, default_transforms)
kg
Applying HeadlinesExtractor:   0%|          | 0/2 [00:00<?, ?it/s]
Applying HeadlineSplitter:   0%|          | 0/2 [00:00<?, ?it/s]
Applying SummaryExtractor:   0%|          | 0/2 [00:00<?, ?it/s]
Applying CustomNodeFilter:   0%|          | 0/8 [00:00<?, ?it/s]
Applying [EmbeddingExtractor, ThemesExtractor, NERExtractor]:   0%|          | 0/18 [00:00<?, ?it/s]
Applying [CosineSimilarityBuilder, OverlapScoreBuilder]:   0%|          | 0/2 [00:00<?, ?it/s]
KnowledgeGraph(nodes: 10, relationships: 19)
We can save and load our knowledge graphs as follows.

kg.save("usecase_data_kg.json")
usecase_data_kg = KnowledgeGraph.load("usecase_data_kg.json")
usecase_data_kg
KnowledgeGraph(nodes: 10, relationships: 19)
Using our knowledge graph, we can construct a "test set generator" - which will allow us to create queries.

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=usecase_data_kg)
However, we'd like to be able to define the kinds of queries we're generating - which is made simple by Ragas having pre-created a number of different "QuerySynthesizer"s.

Each of these Synthetsizers is going to tackle a separate kind of query which will be generated from a scenario and a persona.

In essence, Ragas will use an LLM to generate a persona of someone who would interact with the data - and then use a scenario to construct a question from that data and persona.

from ragas.testset.synthesizers import default_query_distribution, SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer

query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]
❓ Question #1:
What are the three types of query synthesizers doing? Describe each one in simple terms.

Answer:
i. SingleHopSpecificQuerySynthesizer generates direct, specific questions that can be answered from one place in the knowledge (typically one chunk/section). So there is one hop, one source.

ii. MultiHopAbstractQuerySynthesizer generates higher-level "combine ideas" questions that require pulling multiple related concepts (often across sections/docs), but phrased in a more general/abstract way.It blends topics and expects synthesis, not a pinpoint fact.

iii. MultiHopSpecificQuerySynthesizer generates cross-reference questions with explicit anchors (e.g., “Chapter X and Chapter Y…”, or two named topics) that require multiple hops across specific sources to answer. So we have multiple sources here, explicitly tied together.

Finally, we can use our TestSetGenerator to generate our testset!

testset = generator.generate(testset_size=10, query_distribution=query_distribution)
testset.to_pandas()
Generating personas:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Scenarios:   0%|          | 0/3 [00:00<?, ?it/s]
Generating Samples:   0%|          | 0/8 [00:00<?, ?it/s]
user_input	reference_contexts	reference	synthesizer_name
0	World Health Organization what?	[The Mental Health and Psychology Handbook A P...	The World Health Organization is mentioned in ...	single_hop_specifc_query_synthesizer
1	What is COGNITIV BEHAVIORAL THERAPY?	[PART 2: THERAPEUTIC APPROACHES Chapter 4: Cog...	Cognitive Behavioral Therapy is one of the mos...	single_hop_specifc_query_synthesizer
2	How does serotonin relate to mental health and...	[Write letters to or from your future self Jou...	Serotonin is mentioned as a neurotransmitter t...	single_hop_specifc_query_synthesizer
3	How do psychiatrists help with mental health i...	[social interactions How to set and maintain b...	Psychiatrists are medical doctors who can pres...	single_hop_specifc_query_synthesizer
4	How do minerals contribute to overall health a...	[The Personal Wellness Guide A Comprehensive R...	The Personal Wellness Guide emphasizes that mi...	single_hop_specifc_query_synthesizer
5	How can understanding the science of sleep and...	[<1-hop>\n\nPART 5: BUILDING HEALTHY HABITS Ch...	Understanding the science of sleep from Chapte...	multi_hop_specific_query_synthesizer
6	How can Cognitive Behavioral Therapy (CBT) and...	[<1-hop>\n\nPART 3: SLEEP AND RECOVERY Chapter...	Cognitive Behavioral Therapy (CBT) is a widely...	multi_hop_specific_query_synthesizer
7	How can CBT-I help improve sleep and mental he...	[<1-hop>\n\nWrite letters to or from your futu...	CBT-I, or Cognitive Behavioral Therapy for Ins...	multi_hop_specific_query_synthesizer
Abstracted SDG
The above method is the full process - but we can shortcut that using the provided abstractions!

This will generate our knowledge graph under the hood, and will - from there - generate our personas and scenarios to construct our queries.

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)
Applying HeadlinesExtractor:   0%|          | 0/2 [00:00<?, ?it/s]
Applying HeadlineSplitter:   0%|          | 0/2 [00:00<?, ?it/s]
Applying SummaryExtractor:   0%|          | 0/2 [00:00<?, ?it/s]
Applying CustomNodeFilter:   0%|          | 0/8 [00:00<?, ?it/s]
Applying [EmbeddingExtractor, ThemesExtractor, NERExtractor]:   0%|          | 0/18 [00:00<?, ?it/s]
Applying [CosineSimilarityBuilder, OverlapScoreBuilder]:   0%|          | 0/2 [00:00<?, ?it/s]
Generating personas:   0%|          | 0/2 [00:00<?, ?it/s]
Generating Scenarios:   0%|          | 0/3 [00:00<?, ?it/s]
Generating Samples:   0%|          | 0/12 [00:00<?, ?it/s]
dataset.to_pandas()
user_input	reference_contexts	reference	synthesizer_name
0	What are the recommended exercises and strateg...	[The Personal Wellness Guide A Comprehensive R...	The provided context does not include specific...	single_hop_specifc_query_synthesizer
1	What does Stage 2 of sleep involve in the slee...	[PART 3: SLEEP AND RECOVERY Chapter 7: The Sci...	Stage 2 involves a drop in body temperature an...	single_hop_specifc_query_synthesizer
2	What information does Chapter 18 cover regardi...	[PART 5: BUILDING HEALTHY HABITS Chapter 13: T...	Chapter 18 discusses strategies to boost immun...	single_hop_specifc_query_synthesizer
3	How does the World Health Organization define ...	[The Mental Health and Psychology Handbook A P...	According to the World Health Organization, me...	single_hop_specifc_query_synthesizer
4	how can exercise for common problems like lowe...	[<1-hop>\n\nThe Personal Wellness Guide A Comp...	The wellness guide explains that gentle exerci...	multi_hop_abstract_query_synthesizer
5	How can incorporating mindfulness and social c...	[<1-hop>\n\nhour before bed - No caffeine afte...	Incorporating mindfulness and social connectio...	multi_hop_abstract_query_synthesizer
6	How can improving face-to-face interactions an...	[<1-hop>\n\nhour before bed - No caffeine afte...	Improving face-to-face interactions by engagin...	multi_hop_abstract_query_synthesizer
7	How can I improve my emotional intelligence an...	[<1-hop>\n\nhour before bed - No caffeine afte...	To improve emotional intelligence and manage c...	multi_hop_abstract_query_synthesizer
8	how chapter 7 and 17 connect about sleep and h...	[<1-hop>\n\nPART 3: SLEEP AND RECOVERY Chapter...	chapter 7 talks about sleep and recovery, expl...	multi_hop_specific_query_synthesizer
9	H0w c4n I bUild a he4lthy m0rn1ng r0utine (cha...	[<1-hop>\n\nPART 3: SLEEP AND RECOVERY Chapter...	To build a healthy morning routine that improv...	multi_hop_specific_query_synthesizer
10	How can Cognitive Behavioral Therapy (CBT), in...	[<1-hop>\n\nPART 3: SLEEP AND RECOVERY Chapter...	Cognitive Behavioral Therapy (CBT), particular...	multi_hop_specific_query_synthesizer
11	How does Cognitive Behavioral Therapy (CBT) fo...	[<1-hop>\n\nPART 3: SLEEP AND RECOVERY Chapter...	Cognitive Behavioral Therapy for Insomnia (CBT...	multi_hop_specific_query_synthesizer
❓ Question #2:
Ragas offers both an "unrolled" (manual) approach and an "abstracted" (automatic) approach to synthetic data generation. What are the trade-offs between these two approaches? When would you choose one over the other?

Answer:
The unrolled (manual) approach gives you full control over every step of the synthetic data pipeline. You explicitly build the knowledge graph, apply transformations, define the query distribution, and inspect intermediate artifacts (nodes, relationships, summaries, embeddings). This makes it highly transparent and customizable. You can tune how entities are extracted, adjust relationship thresholds, bias toward more multi-hop queries, or debug weak graph connections. The trade-off is complexity: it requires more setup, more code, and deeper understanding of how Ragas constructs test data. It’s ideal when you care about evaluation rigor, need fine-grained control over query types, or are diagnosing RAG weaknesses in production.

The abstracted (automatic) approach wraps the entire workflow into a single high-level call. It automatically builds the knowledge graph, applies default transforms, generates personas and scenarios, and produces the dataset. This is much faster and simpler, which is great for rapid experimentation or early-stage iteration. However, you sacrifice visibility and control. You cannot easily adjust how relationships are formed or precisely shape the query distribution without additional configuration. It’s ideal when you want quick directional evaluation signals or are prototyping before investing in deeper evaluation design.

Summary: - Use abstracted when you want speed and convenience. - Use unrolled when you want control, transparency, and production-grade evaluation tuning.

🏗️ Activity #1: Custom Query Distribution
Modify the query_distribution to experiment with different ratios of query types.

Requirements:
Create a custom query distribution with different weights than the default
Generate a new test set using your custom distribution
Compare the types of questions generated with the default distribution
Explain why you chose the weights you did
Step 1) Define the DEFAULT distribution (from the notebook) Meaning of the default:

50% single-hop specific: easier, direct questions answerable from one place
25% multi-hop abstract: synthesis across concepts, phrased more generally
25% multi-hop specific: cross-references across specific sections/topics
### YOUR CODE HERE ###

# Define a custom query distribution with different weights
# Generate a new test set and compare with the default
from collections import Counter
import pandas as pd

from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)

# ============================================================
# Step 1) Define the DEFAULT distribution (from the notebook)
# ------------------------------------------------------------
# Meaning of the default:
#   - 50% single-hop specific: easier, direct questions answerable from one place
#   - 25% multi-hop abstract: synthesis across concepts, phrased more generally
#   - 25% multi-hop specific: cross-references across specific sections/topics
# ============================================================

default_query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.50),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]
Step 2) Define a CUSTOM distribution (different weights) Why change weights? In real RAG usage, users often ask questions that require:

combining evidence across multiple chunks/sources (multi-hop)
synthesis rather than copying one snippet With only 2 documents, single-hop questions can be "too easy", causing evaluation to overestimate real-world robustness. Custom choice here:
20% single-hop: keep a baseline of easy/direct checks
40% multi-hop abstract: test conceptual synthesis (harder)
40% multi-hop specific: test cross-reference grounding (harder)
# ============================================================
# Step 2) Define a CUSTOM distribution (different weights)
# ------------------------------------------------------------
# Why change weights?
# In real RAG usage, users often ask questions that require:
#   - combining evidence across multiple chunks/sources (multi-hop)
#   - synthesis rather than copying one snippet
#
# With only 2 documents, single-hop questions can be "too easy",
# causing evaluation to overestimate real-world robustness.
#
# Custom choice here:
#   - 20% single-hop: keep a baseline of easy/direct checks
#   - 40% multi-hop abstract: test conceptual synthesis (harder)
#   - 40% multi-hop specific: test cross-reference grounding (harder)
# ============================================================

custom_query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.20),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.40),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.40),
]
Step 3) Generate test sets (DEFAULT vs CUSTOM) for comparison

Use the same testset_size for both, so the mixes are comparable.
For small sizes like 10, the observed percentages can be noisy. Use 20+ for a clearer signal (still fast).
# ============================================================
# Step 3) Generate test sets (DEFAULT vs CUSTOM) for comparison
# ------------------------------------------------------------
# IMPORTANT:
#   - Use the same testset_size for both, so the mixes are comparable.
#   - For small sizes like 10, the observed percentages can be noisy.
#     Use 20+ for a clearer signal (still fast).
# ============================================================

TESTSET_SIZE = 20

default_testset = generator.generate(
    testset_size=TESTSET_SIZE,
    query_distribution=default_query_distribution
)

custom_testset = generator.generate(
    testset_size=TESTSET_SIZE,
    query_distribution=custom_query_distribution
)

# Convert to pandas so we can analyze and compare
default_df = default_testset.to_pandas()
custom_df  = custom_testset.to_pandas()
Generating Scenarios:   0%|          | 0/3 [00:00<?, ?it/s]
Generating Samples:   0%|          | 0/21 [00:00<?, ?it/s]
Generating Scenarios:   0%|          | 0/3 [00:00<?, ?it/s]
Generating Samples:   0%|          | 0/21 [00:00<?, ?it/s]
Step 4) Compare the types of questions generated We do two comparisons: A) Counts by synthesizer_name B) Percentages by synthesizer_name NOTE: The distribution won't match weights perfectly (LLM + sampling + small N),but it should move in the direction we intended.

# ============================================================
# Step 4) Compare the *types* of questions generated
# ------------------------------------------------------------
# We do two comparisons:
#   A) Counts by synthesizer_name
#   B) Percentages by synthesizer_name
#
# NOTE:
# The distribution won't match weights perfectly (LLM + sampling + small N),
# but it should move in the direction we intended.
# ============================================================

def summarize_mix(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a summary table with counts and percentages for each synthesizer."""
    counts = df["synthesizer_name"].value_counts(dropna=False)
    perc = (counts / len(df) * 100).round(1)
    out = pd.DataFrame({
        f"{label}_count": counts,
        f"{label}_pct": perc
    })
    return out

mix_default = summarize_mix(default_df, "default")
mix_custom  = summarize_mix(custom_df, "custom")

mix_comparison = (
    mix_default
    .join(mix_custom, how="outer")
    .fillna(0)
)

# Make counts integers
for col in mix_comparison.columns:
    if col.endswith("_count"):
        mix_comparison[col] = mix_comparison[col].astype(int)

print("=== Synthesizer mix comparison (counts + %) ===")
display(mix_comparison)
=== Synthesizer mix comparison (counts + %) ===
default_count	default_pct	custom_count	custom_pct
synthesizer_name				
multi_hop_abstract_query_synthesizer	5	23.8	8	38.1
multi_hop_specific_query_synthesizer	6	28.6	9	42.9
single_hop_specifc_query_synthesizer	10	47.6	4	19.0
We'll need to provide our LangSmith API key, and set tracing to "true".

Step 5) Show example questions so you can see the difference We'll print a small sample grouped by synthesizer. This gives a qualitative feel:

single-hop: direct fact/section questions
multi-hop abstract: "how do X and Y relate" style
multi-hop specific: "combine chapter A and B" style
# ============================================================
# Step 5) Show example questions so you can *see* the difference
# ------------------------------------------------------------
# We'll print a small sample grouped by synthesizer.
# This gives a qualitative feel:
#   - single-hop: direct fact/section questions
#   - multi-hop abstract: "how do X and Y relate" style
#   - multi-hop specific: "combine chapter A and B" style
# ============================================================

import pandas as pd

# Show full text in cells (no "...")
pd.set_option("display.max_colwidth", None)

# (Optional) widen the notebook cell rendering
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def show_samples(df: pd.DataFrame, label: str, n_per_type: int = 2) -> None:
    print(f"\n=== Sample questions ({label}) ===")
    cols = ["synthesizer_name", "user_input"]
    # Group and show a couple from each type
    for synth, group in df.groupby("synthesizer_name"):
        print(f"\n--- {synth} (showing up to {n_per_type}) ---")
        display(group[cols].head(n_per_type))

show_samples(default_df, "default", n_per_type=2)
show_samples(custom_df, "custom", n_per_type=2)
=== Sample questions (default) ===

--- multi_hop_abstract_query_synthesizer (showing up to 2) ---
synthesizer_name	user_input
10	multi_hop_abstract_query_synthesizer	how stress reduction and mindfulness meditation and yoga help in mental health
11	multi_hop_abstract_query_synthesizer	How can stress reduction be achieved through mindfulness meditation and yoga, especially considering the techniques from MBSR and the importance of mindful living?
--- multi_hop_specific_query_synthesizer (showing up to 2) ---
synthesizer_name	user_input
15	multi_hop_specific_query_synthesizer	How does sleep influence mental health, and what strategies related to sleep can a holistic wellness coach recommend to improve sleep quality and address sleep issues?
16	multi_hop_specific_query_synthesizer	How does slep affect mental health and why is it important for holistc wellnes?
--- single_hop_specifc_query_synthesizer (showing up to 2) ---
synthesizer_name	user_input
0	single_hop_specifc_query_synthesizer	How does the Psychology Handbook contribute to understanding mental health from a holistic perspective?
1	single_hop_specifc_query_synthesizer	What is the significance of the Psychology Handbook for understanding mental health?
=== Sample questions (custom) ===

--- multi_hop_abstract_query_synthesizer (showing up to 2) ---
synthesizer_name	user_input
4	multi_hop_abstract_query_synthesizer	How do sleep hygiene practices, such as maintaining a consistent sleep schedule and creating an optimal sleep environment, support the different stages of sleep like REM and non-REM, and how can these practices help manage sleep problems like insomnia?
5	multi_hop_abstract_query_synthesizer	How does the impact of mental health on physical health, particularly through stress and sleep disturbances, relate to the importance of sleep and its impact on mental health, as discussed in the context of holistic lifestyle practices?
--- multi_hop_specific_query_synthesizer (showing up to 2) ---
synthesizer_name	user_input
12	multi_hop_specific_query_synthesizer	how can CBT and CBT-I help improve mental health through the practices like journaling and mindfulness, especially when dealing with sleep and emotional issues, and what are the benefits of combining these approaches for overall well-being?
13	multi_hop_specific_query_synthesizer	How can CBT and CBT-I help improve mental health through exercise and sleep practices?
--- single_hop_specifc_query_synthesizer (showing up to 2) ---
synthesizer_name	user_input
0	single_hop_specifc_query_synthesizer	Wha is the role of the World Health Organizaton in mental health?
1	single_hop_specifc_query_synthesizer	What CBT means and how it helps with mental health?
Step 6) A single "headline" metric:

"Hardness score" proxy: multi-hop proportion.
This is a crude but useful directional indicator: higher multi-hop % => dataset is harder => stresses RAG more.
# ============================================================
# Step 6) A single "headline" metric:
# ------------------------------------------------------------
# "Hardness score" proxy: multi-hop proportion.
# This is a crude but useful directional indicator:
#   higher multi-hop % => dataset is harder => stresses RAG more.
# ============================================================

def multihop_pct(df: pd.DataFrame) -> float:
    mh = df["synthesizer_name"].str.contains("multi_hop", case=False, na=False).sum()
    return round(mh / len(df) * 100, 1)

print("\n=== Quick hardness proxy ===")
print(f"Default multi-hop %: {multihop_pct(default_df)}")
print(f"Custom  multi-hop %: {multihop_pct(custom_df)}")
=== Quick hardness proxy ===
Default multi-hop %: 52.4
Custom  multi-hop %: 81.0
Step 7) Write-up

# ============================================================
# Step 7) Write-up 
# ============================================================

activity_writeup = f"""
I created a custom query distribution to stress-test the RAG system with harder, more realistic queries.

Default distribution:
- 50% single-hop specific
- 25% multi-hop abstract
- 25% multi-hop specific

Custom distribution:
- 20% single-hop specific
- 40% multi-hop abstract
- 40% multi-hop specific

Why these weights:
With only two overlapping documents, single-hop questions can be too easy and overestimate performance.
Real user questions often require synthesizing across topics (multi-hop) and grounding across multiple sections.
So I increased multi-hop coverage to {multihop_pct(custom_df)}% (vs {multihop_pct(default_df)}% in default) to better test:
1) retrieval recall across multiple relevant chunks, and
2) the model’s ability to synthesize without hallucinating.

Comparison:
The synthesizer mix table (counts + %) shows the custom dataset contains a higher proportion of multi-hop questions,
and the sampled questions illustrate the shift from direct, single-source queries to cross-topic and cross-section questions.
"""
print(activity_writeup)
I created a custom query distribution to stress-test the RAG system with harder, more realistic queries.

Default distribution:
- 50% single-hop specific
- 25% multi-hop abstract
- 25% multi-hop specific

Custom distribution:
- 20% single-hop specific
- 40% multi-hop abstract
- 40% multi-hop specific

Why these weights:
With only two overlapping documents, single-hop questions can be too easy and overestimate performance.
Real user questions often require synthesizing across topics (multi-hop) and grounding across multiple sections.
So I increased multi-hop coverage to 81.0% (vs 52.4% in default) to better test:
1) retrieval recall across multiple relevant chunks, and
2) the model’s ability to synthesize without hallucinating.

Comparison:
The synthesizer mix table (counts + %) shows the custom dataset contains a higher proportion of multi-hop questions,
and the sampled questions illustrate the shift from direct, single-source queries to cross-topic and cross-section questions.

🤝 Breakout Room #2
RAG Evaluation with LangSmith
Task 4: LangSmith Dataset
Now we can move on to creating a dataset for LangSmith!

First, we'll need to create a dataset on LangSmith using the Client!

We'll name our Dataset to make it easy to work with later.

from langsmith import Client
import uuid

client = Client()

dataset_name = f"Use Case Synthetic Data - AIE9 - {uuid.uuid4()}"

langsmith_dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Synthetic Data for Use Cases"
)
We'll iterate through the RAGAS created dataframe - and add each example to our created dataset!

NOTE: We need to conform the outputs to the expected format - which in this case is: question and answer.

for data_row in dataset.to_pandas().iterrows():
  client.create_example(
      inputs={
          "question": data_row[1]["user_input"]
      },
      outputs={
          "answer": data_row[1]["reference"]
      },
      metadata={
          "context": data_row[1]["reference_contexts"]
      },
      dataset_id=langsmith_dataset.id
  )
Basic RAG Chain
Time for some RAG!

rag_documents = docs
To keep things simple, we'll just use LangChain's recursive character text splitter!

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

rag_documents = text_splitter.split_documents(rag_documents)
We'll create our vectorstore using OpenAI's text-embedding-3-small embedding model.

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
As usual, we will power our RAG application with Qdrant!

from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    documents=rag_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="use_case_rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
To get the "A" in RAG, we'll provide a prompt.

from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
As is usual: We'll be using gpt-4.1-mini for our RAG!

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1-mini")
Finally, we can set-up our RAG LCEL chain!

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | rag_prompt | llm | StrOutputParser()
)
rag_chain.invoke({"question" : "What are some recommended exercises for lower back pain?"})
'Recommended exercises for lower back pain include:\n\n- Cat-Cow Stretch: Start on hands and knees, alternate between arching your back up (cat) and letting it sag down (cow). Do 10-15 repetitions.\n- Bird Dog: From hands and knees, extend opposite arm and leg while keeping your core engaged. Hold for 5 seconds, then switch sides. Do 10 repetitions per side.\n- Partial Crunches: Lie on your back with knees bent, cross arms over chest, tighten stomach muscles and raise shoulders off floor. Hold briefly, then lower. Do 8-12 repetitions.\n- Knee-to-Chest Stretch: Lie on your back, pull one knee toward your chest while keeping the other foot flat. Hold for 15-30 seconds, then switch legs.\n- Pelvic Tilts: Lie on your back with knees bent, flatten your back against the floor by tightening abs and tilting pelvis up slightly. Hold for 10 seconds, repeat 8-12 times.'
LangSmith Evaluation Set-up
We'll use OpenAI's GPT-4.1 as our evaluation LLM for our base Evaluators.

eval_llm = ChatOpenAI(model="gpt-4.1")
We'll be using a number of evaluators - from LangSmith provided evaluators, to a few custom evaluators!

from openevals.llm import create_llm_as_judge
from langsmith.evaluation import evaluate

# 1. QA Correctness (replaces LangChainStringEvaluator("qa"))
qa_evaluator = create_llm_as_judge(
    prompt="You are evaluating a QA system. Given the input, assess whether the prediction is correct.\n\nInput: {inputs}\nPrediction: {outputs}\nReference answer: {reference_outputs}\n\nIs the prediction correct? Return 1 if correct, 0 if incorrect.",
    feedback_key="qa",
    model="openai:gpt-4o" ,  # pass your LangChain chat model directly
)

# 2. Labeled Helpfulness (replaces LangChainStringEvaluator("labeled_criteria"))
labeled_helpfulness_evaluator = create_llm_as_judge(
    prompt=(
        "You are assessing a submission based on the following criterion:\n\n"
        "helpfulness: Is this submission helpful to the user, "
        "taking into account the correct reference answer?\n\n"
        "Input: {inputs}\n"
        "Submission: {outputs}\n"
        "Reference answer: {reference_outputs}\n\n"
        "Does the submission meet the criterion? Return 1 if yes, 0 if no."
    ),
    feedback_key="helpfulness",
    model="openai:gpt-4o" ,
)

# 3. Dopeness (replaces LangChainStringEvaluator("criteria"))
dopeness_evaluator = create_llm_as_judge(
    prompt=(
        "You are assessing a submission based on the following criterion:\n\n"
        "dopeness: Is this response dope, lit, cool, or is it just a generic response?\n\n"
        "Input: {inputs}\n"
        "Submission: {outputs}\n\n"
        "Does the submission meet the criterion? Return 1 if yes, 0 if no."
    ),
    feedback_key="dopeness",
    model="openai:gpt-4o" ,
)
Describe what each evaluator is evaluating:

qa_evaluator: This evaluator checks factual correctness. It compares the model’s prediction against the reference answer and determines whether the answer is accurate. It does not care about writing style or tone — only whether the response matches the expected answer. It returns 1 if correct, 0 if incorrect. This measures core QA reliability. So Summary: QA → Is it correct?
labeled_helpfulness_evaluator: This evaluator checks whether the response is actually helpful to the user, taking the reference answer into account. A response might be factually correct but incomplete or poorly structured. This metric evaluates clarity, completeness, and usefulness relative to what the user needs. It returns 1 if helpful, 0 if not helpful. So Summary: Helpfulness → Is it useful?
dopeness_evaluator: This evaluator measures response quality and engagement style. It checks whether the answer is compelling, well-written, and non-generic — not just technically correct. It focuses on tone, richness, and creativity rather than strict correctness. It returns 1 if the response is “dope” (engaging and high-quality), 0 if generic or bland. So Summary: Dopeness → Is it engaging and high-quality?
LangSmith Evaluation
evaluate(
    rag_chain.invoke,
    data=dataset_name,
    evaluators=[
        qa_evaluator,
        labeled_helpfulness_evaluator,
        dopeness_evaluator
    ],
    metadata={"revision_id": "default_chain_init"},
)
View the evaluation results for experiment: 'advanced-shame-28' at:
https://smith.langchain.com/o/2bf8dd81-d1e2-410c-a798-2636bdfe0958/datasets/e027d2f3-0749-4da0-a81e-35a5047090f8/compare?selectedSessions=eee95471-dd8a-4f6b-a1ac-160731fa05bb


0it [00:00, ?it/s]
inputs.question	outputs.output	error	reference.answer	feedback.qa	feedback.helpfulness	feedback.dopeness	execution_time	example_id	id
0	how sleep affects mental health and sleep good or bad	Based on the context provided:\n\nSleep affects mental health in several important ways:\n\n- Sleep deprivation impairs emotional regulation and increases emotional reactivity.\n- Insufficient sleep raises the risk of developing depression by 2-3 times.\n- REM sleep helps process emotional memories and experiences.\n- Chronic insomnia is both a symptom and a risk factor for anxiety disorders.\n- Sleep loss reduces the ability to cope with stress.\n\nTherefore, good sleep is essential for maintaining good mental health, while poor sleep can trigger or worsen mental health conditions. Addressing sleep issues is frequently a critical component of mental health treatment.	None	Sleep and mental health are closely connected; poor sleep can worsen mental health conditions like depression and anxiety, while mental health issues can disrupt sleep. The context explains that sleep deprivation impairs emotional regulation, increases stress hormones, and reduces the ability to cope with stress. It also states that insufficient sleep raises the risk of depression by 2-3 times and that chronic insomnia is both a symptom and a risk factor for anxiety disorders. Good sleep quality, achieved through practices like maintaining a consistent sleep schedule, creating a relaxing bedtime routine, and optimizing the sleep environment, is essential for mental well-being. Therefore, sleep quality directly influences mental health, with good sleep promoting better emotional regulation and resilience, while poor sleep can contribute to mental health problems.	True	True	False	4.179342	9d5b472a-8eb9-4da7-9e30-db4bdfa48259	019c67d8-9a8e-7cf2-8325-a8188dc14a94
1	How does sleep influence mental health, and what strategies can a holistic wellness coach recommend to improve sleep quality and address sleep issues related to mental health?	Sleep influences mental health through a bidirectional relationship: poor sleep can trigger or worsen mental health conditions, while mental health problems often disrupt sleep. Specifically, sleep deprivation impairs emotional regulation and increases reactivity, raises the risk of depression by 2-3 times, and reduces the ability to cope with stress. REM sleep plays an important role in processing emotional memories and experiences. Chronic insomnia is both a symptom and risk factor for anxiety disorders. Sleep disorders such as insomnia, hypersomnia (excessive sleeping often linked to depression), nightmares (associated with PTSD and anxiety), and sleep apnea (linked to depression and cognitive impairment) are commonly connected to mental health conditions.\n\nTo improve sleep quality and address sleep issues related to mental health, a holistic wellness coach can recommend these strategies:\n\n1. **Sleep Hygiene Practices**:\n   - Maintain a consistent sleep schedule, including weekends.\n   - Create a relaxing bedtime routine such as reading, gentle stretching, or taking a warm bath.\n   - Keep the bedroom cool, dark, and quiet.\n   - Limit screen exposure 1-2 hours before bed.\n   - Avoid caffeine after 2 PM.\n   - Exercise regularly but not too close to bedtime.\n   - Limit alcohol and heavy meals before bedtime.\n\n2. **Cognitive Behavioral Therapy for Insomnia (CBT-I) Components**:\n   - Sleep restriction: Limit time in bed to actual sleep time.\n   - Stimulus control: Use the bed only for sleep and intimacy.\n   - Cognitive therapy: Address beliefs about sleep.\n   - Sleep hygiene education: Optimize the sleep environment.\n   - Relaxation training: Use progressive muscle relaxation and breathing exercises.\n\n3. **Incorporate Regular Exercise**:\n   - Regular physical activity improves sleep quality and duration, reduces anxiety and depression symptoms, and promotes overall mental well-being.\n\nBy combining these recommendations, a holistic wellness coach can effectively support clients in improving sleep quality and addressing sleep-related mental health issues.	None	Sleep plays a crucial role in mental health by repairing tissues, consolidating memories, and regulating hormones, with insufficient or poor-quality sleep impairing emotional regulation and increasing the risk of depression and anxiety. Chronic insomnia, sleep deprivation, and disrupted sleep cycles can exacerbate mental health conditions, while good sleep hygiene practices—such as maintaining a consistent sleep schedule, creating a relaxing bedtime routine, optimizing the sleep environment, and managing caffeine and alcohol intake—are essential strategies to improve sleep quality. A holistic wellness coach can recommend techniques like establishing regular sleep routines, practicing relaxation exercises, and addressing underlying stress or emotional issues to support better sleep and, consequently, enhance overall mental well-being.	True	True	True	7.015868	1c01a860-6a50-46a3-8276-4f3cfd40a79b	019c67d8-d67f-7380-97a1-b969418e1c9a
2	How does CBT-I relate to CBT in improving mental health, especially with sleep and thoughts?	CBT-I (Cognitive Behavioral Therapy for Insomnia) is a specialized form of CBT that focuses specifically on treating chronic insomnia. Like general CBT, CBT-I addresses thoughts and behaviors, but it targets those related to sleep. CBT-I components include sleep restriction, stimulus control, cognitive therapy addressing beliefs about sleep, sleep hygiene education, and relaxation training. CBT-I is the recommended first-line treatment for chronic insomnia and is more effective than medication in the long term without side effects.\n\nMore broadly, CBT works by identifying and changing negative thought patterns and behaviors that contribute to emotional distress, showing strong evidence for treating conditions including insomnia. CBT emphasizes the interconnectedness of thoughts, feelings, and behaviors and uses techniques like cognitive restructuring to challenge distorted thoughts.\n\nTherefore, CBT-I relates to CBT in that it applies core CBT principles—addressing negative thoughts and changing behaviors—but focuses these strategies on sleep-related issues, thereby improving both sleep and mental health through better management of thoughts and behaviors around sleep.	None	CBT-I is a specific form of Cognitive Behavioral Therapy focused on treating insomnia by addressing sleep habits and thoughts about sleep, and it is considered the gold standard for sleep issues. General CBT aims to identify and change negative thought patterns and behaviors that contribute to emotional distress. Both approaches involve structured exercises like cognitive restructuring and behavioral activation, and they work together to improve mental health by modifying unhelpful thoughts and promoting healthier behaviors, including better sleep and emotional regulation.	True	True	False	4.403529	32c56b32-406e-4c15-8df7-1b52ee2e64e7	019c67d9-2120-7231-8735-548567253d81
3	how CBT and CBT-I help with mental health and sleep problems like insomnia and anxiety	Based on the provided context:\n\nCBT (Cognitive Behavioral Therapy) helps with mental health problems, including anxiety and insomnia, by addressing the interconnectedness of thoughts, feelings, and behaviors. It identifies and challenges negative thinking patterns and teaches new coping skills through structured exercises and homework. Common CBT techniques include cognitive restructuring (challenging distorted thoughts), behavioral activation (scheduling meaningful activities), exposure therapy (facing feared situations gradually), thought records, and problem-solving skills.\n\nCBT-I (Cognitive Behavioral Therapy for Insomnia) is the recommended first-line treatment for chronic insomnia and is more effective than medication in the long term with no side effects. CBT-I includes specific components such as sleep restriction (limiting time in bed to actual sleep time), stimulus control (using the bed only for sleep and intimacy), cognitive therapy (addressing beliefs about sleep), sleep hygiene education (optimizing the sleep environment), and relaxation training (progressive muscle relaxation and breathing exercises).\n\nBoth CBT and CBT-I improve sleep and mental health by challenging and changing unhelpful thoughts and behaviors related to sleep and anxiety. Addressing sleep problems like insomnia also helps reduce risks associated with poor sleep, such as impaired emotional regulation, increased anxiety, and depression. Therefore, CBT and CBT-I help by treating underlying cognitive and behavioral factors contributing to insomnia and anxiety, improving emotional regulation, and enhancing coping with stress.	None	CBT helps mental health by changing negative thoughts and behaviors, which can treat depression, anxiety, and other issues. CBT-I, a specific form of CBT, is the first-line treatment for chronic insomnia, using techniques like sleep restriction, stimulus control, and relaxation training. Combining these approaches addresses both mental health and sleep problems, as CBT targets negative thinking patterns linked to anxiety and depression, while CBT-I improves sleep quality, which in turn supports better emotional regulation and reduces stress.	True	True	False	5.779407	bb9e88d0-0be8-473f-bf0c-1183092eb0a7	019c67d9-7878-7292-b3a8-c5131fefd6ad
4	how build healthy morning routines for wellness and productivity using habit formation and building routines	To build healthy morning routines for wellness and productivity using habit formation and building routines, you can follow these steps based on the provided context:\n\n1. **Start small and be patient:** Commit to small initial actions (e.g., just 2 minutes) and understand that habits take an average of 66 days to form.\n\n2. **Use habit stacking:** Attach new morning habits to existing ones to create a seamless routine.\n\n3. **Make habits obvious and easy:** Design your morning routine so that each habit is clear and simple to do, which helps in consistent practice.\n\n4. **Track your progress:** Keep a record of your routine to stay motivated and aware of your habit formation.\n\n5. **Sample wellness morning routine to boost productivity and wellness:**\n   - Wake at a consistent time.\n   - Drink a glass of water immediately.\n   - Engage in 5-10 minutes of stretching or light movement.\n   - Eat a healthy breakfast.\n   - Practice brief mindfulness or journaling.\n   - Review your goals and priorities for the day.\n\n6. **Avoid unhelpful morning habits:** Do not check your phone immediately upon waking, skip breakfast, hit snooze multiple times, rush without preparation, or start the day with negative news.\n\n7. **Understand the habit loop:** Use the cue-routine-reward model to reinforce habits. For example, the cue may be waking up, the routine is drinking water and stretching, and the reward is feeling refreshed and energized.\n\nBy combining these habit formation principles with the sample morning routine, you can build a consistent, wellness-focused, and productive start to your day.	None	building healthy morning routines for wellness and productivity involves understanding habit formation by starting small, attaching new habits to existing ones, and tracking progress. A good morning routine can include waking at a consistent time, drinking water, stretching, having a healthy breakfast, practicing mindfulness or journaling, and reviewing goals. Avoid habits like checking your phone immediately or rushing without preparation. Combining these strategies helps create effective routines that boost energy and mood throughout the day.	True	True	True	6.365402	3649c71d-c2fa-47bb-911c-9790cf6b8f35	019c67d9-bfa9-7821-8772-4abd98e1da28
5	H0w can I plaan my meels to make shure I get enuff nutriants and stay hydrayed, espeshully when I have a busy sked and need to keep my diet balenced with good water intake and signs of dehydraytion in mind?	To plan your meals effectively and ensure you get enough nutrients and stay hydrated, especially with a busy schedule, follow these steps based on the provided context:\n\n1. **Weekly Meal Planning Steps:**\n   - Review your schedule for the week and identify busy days.\n   - Choose 3-4 main dishes that can provide leftovers.\n   - Plan for healthy snacks to avoid impulse eating.\n   - Create a detailed shopping list organized by store section.\n   - Prep ingredients in advance when possible.\n\n2. **Nutrient-Rich Meal Choices:**\n   - Include a variety of colorful fruits and vegetables for vitamins.\n   - Incorporate minerals by eating whole foods, dairy, and leafy greens.\n   - Focus on balanced meals with complex carbohydrates (whole grains, fruits, vegetables), proteins (lean meats, fish, eggs, legumes, dairy), and healthy fats (olive oil, avocados, nuts, fatty fish).\n   - Consider adding immune-boosting nutrients like Vitamin C (citrus fruits, bell peppers, strawberries), Vitamin D (sunlight, fatty fish, fortified foods), Zinc (oysters, beef, pumpkin seeds, chickpeas), and Vitamin E (nuts, seeds, spinach).\n\n3. **Hydration Tips:**\n   - Drink about 8 glasses (64 ounces) of water daily, adjusting based on activity level, climate, and body size.\n   - Start your day with a glass of water.\n   - Carry a reusable water bottle throughout the day.\n   - Set reminders on your phone to drink water regularly.\n   - Eat water-rich foods like cucumbers, watermelon, and oranges.\n   - Drink a glass of water before each meal.\n\n4. **Recognize Signs of Dehydration:**\n   - Watch for dark yellow urine.\n   - Be aware of fatigue and low energy.\n   - Note any headaches, dry mouth, dry skin, or dizziness.\n\nBy organizing your meals and snacks ahead, focusing on nutrient-dense foods, and maintaining consistent hydration habits with water and water-rich foods, you can keep your diet balanced and stay well-hydrated even during busy days.	None	To plaan your meels for optimal nutriants and stay hydrayed, start by reviewing your weekly sked and choosing main dishez that can be prepped in advans. Incorporate a balenced diet with complex carbs, lean proteens, and healthy fats, and include water-rich foods like cucumbers and watermelons. Remember to drink at least 8 glasses of water daily, and watch for signs of dehydraytion such as dark urine or headaches. Using a detailed shopping list and prepping ingredients ahead can help you stick to your plan, even on busy days, ensuring you stay nourished and hydrayed.	True	True	True	7.359522	27d3ecd1-7b75-4f78-9940-cefc5f568d83	019c67da-0738-7d93-96fe-61bfed0c3ac3
6	How can stress reduction be achieved through deep breathing and muscle relaxation in daily practice?	Stress reduction through deep breathing and muscle relaxation in daily practice can be achieved by using immediate relief techniques such as deep breathing—inhale for 4 counts, hold for 4, exhale for 4—and progressive muscle relaxation, which involves tensing and then releasing muscle groups progressively from toes to head. Incorporating these practices regularly helps to calm the nervous system, reduce physical tension, and promote relaxation, thereby managing stress effectively over time.	None	Stress reduction can be achieved through daily practices such as deep breathing, muscle relaxation, grounding, walking, and listening to calming music, which help promote relaxation and emotional well-being.	True	True	False	2.768839	9412076c-3bc7-4d1c-9d17-960fb331ad62	019c67da-6a10-7e20-ab0d-c33c85ff6d54
7	Factors influence mental health and sleep stages how they connect?	Factors that influence mental health include biological factors (such as genetics, brain chemistry, and hormones), life experiences (like trauma or significant life changes), family history of mental health conditions, social environment (relationships and support systems), physical health (including chronic illness, sleep quality, and nutrition), and lifestyle choices (exercise habits, substance use, and stress management).\n\nRegarding sleep stages and their connection to mental health, REM sleep plays an important role in processing emotional memories and experiences. Poor sleep, including sleep deprivation and chronic insomnia, can impair emotional regulation, increase reactivity, and heighten the risk of mental health conditions like depression and anxiety. Sleep disorders such as insomnia, hypersomnia, nightmares, and sleep apnea are also associated with various mental health conditions.\n\nIn summary, sleep quality and stages like REM sleep are closely linked to emotional processing and mental health stability, while various factors influence both mental health and sleep quality, demonstrating their interconnectedness.	None	Factors influencing mental health include biological aspects, life experiences, family history, social environment, physical health, and lifestyle choices. Sleep occurs in stages like REM and non-REM, which are crucial for mental well-being. Poor sleep can worsen mental health, and mental health issues can disrupt sleep stages, showing their connection.	True	True	False	4.261696	b0390ab6-043a-4e80-bbb0-356e7abb932a	019c67da-9e9f-77a3-8221-95ec5a48b826
8	How do Licensed Clinical Social Workers help with setting boundaries and managing digital mental health issues?	I don't know.	None	Licensed Clinical Social Workers provide therapy and connect clients with resources, which can include guidance on setting and maintaining boundaries, such as identifying limits, communicating clearly, and enforcing them consistently. They also address digital mental health challenges by helping individuals develop strategies like setting time limits for social media, scheduling digital detoxes, and practicing mindful technology use to improve psychological well-being.	False	False	False	1.017200	88ba6507-a111-4cf5-a346-b1187e186912	019c67da-e146-7822-866a-2de5586e112b
9	How does exercise help with mental health according to the context?	According to the context, exercise helps with mental health by:\n\n- Releasing endorphins and enkephalins, which are natural mood elevators  \n- Reducing cortisol and adrenaline, which are stress hormones  \n- Increasing serotonin and norepinephrine production  \n- Promoting growth of new brain cells (neurogenesis)  \n- Improving sleep quality and duration  \n- Providing a sense of accomplishment and self-efficacy  \n- Offering social interaction opportunities  \n\nAdditionally, exercise increases neurotransmitter levels, promotes neuroplasticity, reduces inflammation, reduces depression risk by 25-30%, is as effective as medication for mild to moderate depression, reduces anxiety symptoms by approximately 20%, improves cognitive function, and may reduce dementia risk. Regular exercise releases endorphins and reduces symptoms of depression and anxiety.	None	Exercise affects the brain in multiple beneficial ways, including increasing neurotransmitter levels, promoting neuroplasticity, and reducing inflammation. It releases endorphins and enkephalins (natural mood elevators), reduces cortisol and adrenaline (stress hormones), increases serotonin and norepinephrine production, promotes growth of new brain cells (neurogenesis), improves sleep quality and duration, provides a sense of accomplishment and self-efficacy, and offers social interaction opportunities.	True	True	True	3.144251	4da5dd8e-aa17-4e14-bbc6-07537ee4157e	019c67db-0d00-7102-a012-00714a6a88a2
10	What is the significance of the University of Massachusetts Medical Center in the context of mindfulness-based therapies?	The University of Massachusetts Medical Center is significant because it is where Mindfulness-Based Stress Reduction (MBSR) was developed by Jon Kabat-Zinn.	None	The University of Massachusetts Medical Center developed Mindfulness-Based Stress Reduction (MBSR), an 8-week program that teaches mindfulness meditation and yoga to reduce stress and improve well-being.	True	True	False	2.108374	2299a097-2d76-4bb7-af81-40711c32e54a	019c67db-4ef5-7580-9da5-a47cdc0f3483
11	As a Holistic Wellness Coach, how can understanding the concept of mental health contribute to guiding individuals toward achieving overall well-being through practical strategies and mindful living?	Understanding the concept of mental health helps a Holistic Wellness Coach guide individuals toward overall well-being by recognizing that mental health encompasses emotional, psychological, and social well-being, affecting how people think, feel, and act at every stage of life. The coach can emphasize practical strategies such as mindfulness exercises, physical activity, quality sleep, nutritious eating, social connection, and self-care practices like journaling and cultivating a growth mindset. These strategies help individuals cope with daily stresses, improve emotional regulation, and foster resilience. Additionally, being aware of the mind-body connection allows the coach to address how mental and physical health influence each other, promoting a balanced approach. Through mindful living and encouraging self-compassion, gratitude, and asking for help when needed, the coach can support clients in achieving productive, fulfilling lives and contributing positively to their communities.	None	Mental health encompasses our emotional, psychological, and social well-being, affecting how we think, feel, and act. It is important at every stage of life and exists on a spectrum, with everyone's experience being unique. Factors contributing to mental health include biological factors, life experiences, family history, social environment, physical health, and lifestyle choices. Good mental health is not merely the absence of mental illness but involves realizing one's abilities, coping with normal stresses, working productively, and contributing to the community. Understanding these aspects allows a Holistic Wellness Coach to guide individuals in adopting lifestyle habits that promote mental well-being, such as managing stress, fostering supportive relationships, and maintaining physical health through mindful practices.	True	True	True	3.443457	8633b70e-ba3d-4866-b065-35be09b7073f	019c67db-87b5-7cc0-b348-d4c2710230ae
Dope-ifying Our Application
We'll be making a few changes to our RAG chain to increase its performance on our SDG evaluation test dataset!

Include a "dope" prompt augmentation
Use larger chunks
Improve the retriever model to: text-embedding-3-large
Let's see how this changes our evaluation!

DOPENESS_RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.

If you cannot answer the question based on the context - you must say "I don't know".

Make your answer rad, ensure high levels of dopeness. Do not be generic, or give generic responses.

Context: {context}
Question: {question}
"""

dopeness_rag_prompt = ChatPromptTemplate.from_template(DOPENESS_RAG_PROMPT)
rag_documents = docs
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 50
)

rag_documents = text_splitter.split_documents(rag_documents)
❓ Question #3:
Why would modifying our chunk size modify the performance of our application?

Answer:
Modifying chunk size changes how the documents are broken up before being embedded and stored in the vector database — and that directly affects retrieval quality.

In RAG, the retriever does not search entire documents. It searches chunks. So chunk size determines the unit of retrieval.

If chunks are too small, important context may be split across multiple chunks. The retriever might return only part of the needed information, forcing the LLM to answer with incomplete context. This can reduce QA correctness and increase “I don’t know” responses. However, smaller chunks can improve precision because embeddings are more focused and semantically tight.

If chunks are too large, each chunk may contain multiple unrelated ideas. This can dilute embedding quality, making retrieval less precise. Large chunks can also introduce irrelevant text into the prompt, increasing noise and token usage. But they help when answers require broader context or multi-hop reasoning within the same chunk.

So chunk size affects: i. Retrieval precision (how focused embeddings are) ii. Context completeness (whether all needed info fits in one chunk) iii. Prompt length and token cost iv. Multi-hop reasoning success

In our experiment, increasing chunk size from 500 → 1000 likely improved performance because multi-hop relationships (e.g., sleep + mental health connections) were more likely to exist within a single retrieved chunk, reducing fragmentation and improving answer grounding.

Summary: Chunk size changes the granularity of knowledge retrieval, and retrieval quality directly controls RAG performance.

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
❓ Question #4:
Why would modifying our embedding model modify the performance of our application?

Answer:
Changing the embedding model changes how the text is represented in vector space and retrieval quality depends entirely on that representation.

In RAG, embeddings determine how similar two pieces of text appear to the vector database. If the embedding model captures semantic meaning well, related concepts (even if phrased differently) will be close together in vector space. If it does not, retrieval will miss relevant chunks or retrieve weakly related ones.

When we moved from text-embedding-3-small to text-embedding-3-large, we likely improved: i. Semantic resolution – better understanding of nuanced relationships (e.g., sleep and mental health connections). ii. Handling of paraphrases – better matching even when the question wording differs from the document. iii. Multi-hop retrieval quality – better capture of cross-topic relationships. iv. Signal-to-noise ratio – more accurate similarity scoring.

Larger embedding models typically: - Produce higher-dimensional vectors - Capture richer semantic structure - Reduce retrieval errors - Improve recall of relevant chunks

Since RAG performance depends on retrieving the right context, and embeddings control retrieval, improving embeddings often directly improves QA correctness and helpfulness.

Summary: Embedding models shape the geometry of my knowledge space.Better geometry → better retrieval → better answers.

from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    documents=rag_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="Use Case RAG Docs"
)
retriever = vectorstore.as_retriever()
Setting up our new and improved DOPE RAG CHAIN.

dopeness_rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | dopeness_rag_prompt | llm | StrOutputParser()
)
Let's test it on the same output that we saw before.

dopeness_rag_chain.invoke({"question" : "How can I improve my sleep quality?"})
"Yo, ready to level up your sleep game to legendary status? Here's the ultimate slumber blueprint straight from the sleep sages:\n\n1. Lock in a steady sleep schedule — wake and sleep at the same times, no slacking, even on wild weekends. Your body loves routine vibes.\n2. Craft a chill bedtime ritual: think cozy reading, gentle stretches, or a warm bath to signal to your brain it’s time to power down.\n3. Transform your sleep lair: keep the room cool (65-68°F or 18-20°C), pitch-black with blackout curtains or a sleek sleep mask, and dead silent or smooth with whispering white noise.\n4. Ditch screens 1-2 hours before bed. Blue light is the enemy of your melatonin mojo.\n5. Drop caffeine after 2 PM like it’s hot lava — no more jittery nights.\n6. Get moving daily, but keep workouts away from bedtime so you don’t rev your engine too late.\n7. Avoid alcohol and heavy meals before hitting the hay — they mess with your sleep cycles.\n8. Invest in top-tier comfort: a legit mattress and pillows that make you feel like a cloud.\n\nStick to this sleep hygiene like a boss, and you’ll be catching deep, dreamy waves (hello Stage 3 restorative sleep and REM memory magic) like a pro. Sweet dreams, sleep warrior! 🌙✨"
Finally, we can evaluate the new chain on the same test set!

evaluate(
    dopeness_rag_chain.invoke,
    data=dataset_name,
    evaluators=[
        qa_evaluator,
        labeled_helpfulness_evaluator,
        dopeness_evaluator
    ],
    metadata={"revision_id": "dopeness_rag_chain"},
)
View the evaluation results for experiment: 'artistic-expansion-42' at:
https://smith.langchain.com/o/2bf8dd81-d1e2-410c-a798-2636bdfe0958/datasets/e027d2f3-0749-4da0-a81e-35a5047090f8/compare?selectedSessions=390026a4-236b-481e-ba2c-5ee30cc1f92d


0it [00:00, ?it/s]
inputs.question	outputs.output	error	reference.answer	feedback.qa	feedback.helpfulness	feedback.dopeness	execution_time	example_id	id
0	how sleep affects mental health and sleep good or bad	Yo, here’s the straight-up vibe on how sleep plays with mental health — it’s like a wild two-way street. When you’re sleep-deprived, your emotional control goes on vacation, making you super reactive and moody. Lack of sleep can jack up your risk of depression by 2 to 3 times—yeah, that’s huge. REM sleep? It’s the emotional memory processor, helping you sort through feelings like a boss. Chronic insomnia? That’s both a warning sign and a spark for anxiety disorders. Plus, losing sleep messes with your stress-handling superpowers. So bottom line: bad sleep blows for your brain and mood, but when you lock in good quality sleep, you’re basically turbocharging your mental health, keeping depression, anxiety, and stress in check like a champ. Sleep ain't just rest—it’s the ultimate mental health hack.	None	Sleep and mental health are closely connected; poor sleep can worsen mental health conditions like depression and anxiety, while mental health issues can disrupt sleep. The context explains that sleep deprivation impairs emotional regulation, increases stress hormones, and reduces the ability to cope with stress. It also states that insufficient sleep raises the risk of depression by 2-3 times and that chronic insomnia is both a symptom and a risk factor for anxiety disorders. Good sleep quality, achieved through practices like maintaining a consistent sleep schedule, creating a relaxing bedtime routine, and optimizing the sleep environment, is essential for mental well-being. Therefore, sleep quality directly influences mental health, with good sleep promoting better emotional regulation and resilience, while poor sleep can contribute to mental health problems.	True	True	True	3.550773	9d5b472a-8eb9-4da7-9e30-db4bdfa48259	019c67e7-435b-7d42-aef7-1ec804a4a3b4
1	How does sleep influence mental health, and what strategies can a holistic wellness coach recommend to improve sleep quality and address sleep issues related to mental health?	Alright, let's crank up the dopeness on this sleep-mental health synergy! 🎧🛌\n\n**How Sleep Influences Mental Health:**  \nSleep is not just a pit stop—it’s a full-on mental health mechanic. Poor sleep can jam up your emotional regulation, making you more reactive and vulnerable, like your brain’s on a rollercoaster without seatbelts. Sleep deprivation *literally* doubles to triples your risk of depression, while REM sleep acts like your emotional memory processor, helping you digest the feels. Chronic insomnia isn’t just a symptom—it’s a trigger and fuel for anxiety disorders. Plus, losing out on sleep tanks your stress-coping superpowers, leaving you more exposed to mental health battles.\n\n**Strategies a Holistic Wellness Coach Can Drop to Boost Sleep & Mental Vibes:**  \n1. **Sleep Hygiene Mastery:** Nailing a rock-solid routine is key—keep your sleep schedule locked tight (yes, even weekends), dim those screens 1-2 hours before bedtime, nix caffeine after 2 PM, and dodge heavy meals or booze late at night. Temperature magic? Keep your room chill between 65-68°F, pitch-black, and whisper-quiet with blackout curtains or white noise.\n\n2. **Pre-bedtime Rituals:** Flow into sleep mode with relaxing moves—reading, gentle stretching, warm baths, or breathing exercises to prep your mind and body for dreamland.\n\n3. **Cognitive Behavioral Therapy for Insomnia (CBT-I):** This is the heavyweight champ recommended for chronic insomnia—far better than meds in the long haul. It’s a combo of:  \n   - **Sleep restriction** (only stay in bed as long as you actually sleep)  \n   - **Stimulus control** (make your bed a sleep-and-intimacy zone only—no scrolling or stress-baking there)  \n   - **Cognitive therapy** to rewire beliefs stressing your sleep  \n   - **Sleep hygiene education** to optimize your environment  \n   - **Relaxation training** like progressive muscle relaxation and breathwork to chill your nervous system.\n\n4. **Movement and Fresh Air:** Regular exercise (at least 150 mins of moderate activity weekly) amps up mood and sleep quality, but don’t crush it right before bed. Bonus: Outdoor green exercise floods your brain with extra serotonin and zen.\n\nBy weaving these killer strategies into your life, you’re not just chasing better sleep—you’re locking in mental resilience like a boss. Sleep is the secret sauce to mental clarity, mood stability, and stress mastery. So, own your night, own your mind—rest hard, rise unstoppable. 🌙✨	None	Sleep plays a crucial role in mental health by repairing tissues, consolidating memories, and regulating hormones, with insufficient or poor-quality sleep impairing emotional regulation and increasing the risk of depression and anxiety. Chronic insomnia, sleep deprivation, and disrupted sleep cycles can exacerbate mental health conditions, while good sleep hygiene practices—such as maintaining a consistent sleep schedule, creating a relaxing bedtime routine, optimizing the sleep environment, and managing caffeine and alcohol intake—are essential strategies to improve sleep quality. A holistic wellness coach can recommend techniques like establishing regular sleep routines, practicing relaxation exercises, and addressing underlying stress or emotional issues to support better sleep and, consequently, enhance overall mental well-being.	True	True	True	13.226648	1c01a860-6a50-46a3-8276-4f3cfd40a79b	019c67e7-83a3-7f92-9370-0bc596e03837
2	How does CBT-I relate to CBT in improving mental health, especially with sleep and thoughts?	Alright, here’s the legendary scoop on how CBT-I and CBT tag-team to boss up your mental health, especially with sleep and thoughts:\n\nCBT (Cognitive Behavioral Therapy) is the mental ninja that rewires your brain by linking your thoughts, feelings, and behaviors. It hunts down those pesky negative thinking patterns and slams them with cognitive restructuring, behavioral activation, and other solid moves. This mental kung fu is a powerhouse against depression, anxiety, PTSD, and also insomnia.\n\nEnter CBT-I—the sleep specialist within the CBT universe. It’s like the MVP for chronic insomnia, known to be more lit than meds for long-term results and totally side-effect free. CBT-I zooms in on sleep-specific tactics: restricting your time in bed to actual sleep time, using your bed solely for sleep and intimacy (so no Netflix marathons until 3 AM), flipping your sleep beliefs with cognitive therapy, leveling up your sleep hygiene, and mastering relaxation training.\n\nHow do these two crew together? CBT lays the mental scaffolding by challenging distorted thoughts and teaching new coping skills, while CBT-I hones in on the sleep angle—because sleep and mental health are locked in a fierce two-way dance. Bad sleep fires up emotional chaos and mood dips, while mental health struggles crank up insomnia levels.\n\nTogether, they strike like the ultimate dream team: CBT resets your thought patterns and mental habits, while CBT-I recalibrates your sleep rituals and mindset. This combo not only tackles insomnia head-on but also smooths out the mental ripples caused by sleeplessness—giving your emotional regulation a serious upgrade and making you ready to crush stress like a boss.\n\nIn short, CBT-I is the specialized sidekick of CBT that targets sleep’s tight grip on mental health, integrating cognitive therapy and behavioral changes to reboot both your dreams and your day-to-day vibe. Boom.	None	CBT-I is a specific form of Cognitive Behavioral Therapy focused on treating insomnia by addressing sleep habits and thoughts about sleep, and it is considered the gold standard for sleep issues. General CBT aims to identify and change negative thought patterns and behaviors that contribute to emotional distress. Both approaches involve structured exercises like cognitive restructuring and behavioral activation, and they work together to improve mental health by modifying unhelpful thoughts and promoting healthier behaviors, including better sleep and emotional regulation.	True	True	True	4.993941	32c56b32-406e-4c15-8df7-1b52ee2e64e7	019c67e7-f299-7a01-b3a6-d72ba70d8e3b
3	how CBT and CBT-I help with mental health and sleep problems like insomnia and anxiety	Alright, let’s crank up the dopeness on how CBT and its heavyweight sibling CBT-I slam dunk mental health and sleep woes like insomnia and anxiety. 🎯\n\n**CBT (Cognitive Behavioral Therapy)** is the brain’s personal trainer for thoughts, feelings, and behaviors—showing them who's boss. It’s like hacking your mental code by identifying and smashing those pesky negative thinking patterns that fuel anxiety and depression. Through badass techniques like **cognitive restructuring** (flipping distorted thoughts on their head), **behavioral activation** (injecting life with rewarding activities), and **exposure therapy** (facing fears head-on), CBT rewires your brain’s response to stress and emotional chaos.\n\nNow, zoom into **CBT-I (Cognitive Behavioral Therapy for Insomnia)**, the gold standard MVP for chronic insomnia. This specialized form of CBT targets the sleep beast directly—with moves like:\n\n- **Sleep restriction:** Tightening the time in bed to match actual sleep time, a slick way to boost sleep efficiency.\n- **Stimulus control:** Training your brain to link bed only with sleep and intimacy, nixing those distracting Netflix marathons or late-night worries.\n- **Cognitive therapy:** Flipping negative beliefs about sleep that keep you tossing and turning.\n- **Sleep hygiene education & relaxation training:** Crafting the ultimate sleep-friendly environment and chilling out with breathing and muscle relaxation.\n\nHere’s the mic drop: **CBT-I is more effective long-term than meds** and skips all those nasty side effects. Plus, since sleep and mental health vibe in an epic two-way flow, crushing insomnia with CBT-I also brightens emotional regulation and slashes anxiety risk.\n\nIn short, CBT dismantles the mental storms fueling anxiety and depression, while CBT-I tackles sleep struggles from the roots—turning restless nights into power-packed Zzzs and mental chaos into calm vibes. Rock that brain-body balance like a true boss! 🤘😴🔥	None	CBT helps mental health by changing negative thoughts and behaviors, which can treat depression, anxiety, and other issues. CBT-I, a specific form of CBT, is the first-line treatment for chronic insomnia, using techniques like sleep restriction, stimulus control, and relaxation training. Combining these approaches addresses both mental health and sleep problems, as CBT targets negative thinking patterns linked to anxiety and depression, while CBT-I improves sleep quality, which in turn supports better emotional regulation and reduces stress.	True	True	True	4.851347	bb9e88d0-0be8-473f-bf0c-1183092eb0a7	019c67e8-35ae-7a33-aaea-c7f45568bb7c
4	how build healthy morning routines for wellness and productivity using habit formation and building routines	Alright, let's crush that morning routine game with some habit-formation magic straight from the wellness vault:\n\n1. **Cue it up:** Wake at the same time every day—that's your morning bell ringing. Consistency sets the groove.\n\n2. **Start small, start strong:** Commit to just 2 minutes of something, like gentle stretching or mindful breathing. Small wins build unstoppable momentum.\n\n3. **Stack your habits:** Attach new morning moves to what you already do—right after you drink that hydrating glass of water, slide into 5-10 minutes of light movement or journaling. Habit stacking = habit hacking.\n\n4. **Make it obvious & easy:** Keep your yoga mat, journal, or water bottle where you can see and grab them without thinking. The less friction, the better.\n\n5. **Reward yourself:** Feel that sense of calm or energized buzz? Let that be your brain’s jackpot, reinforcing the routine with dopamine.\n\n6. **Mindfulness check-in:** Toss in a 5-minute mindfulness practice or journaling to center your vibe and set daily goals. It’s like fueling your mental jetpack.\n\n7. **Track progress:** Use a habit tracker or just a simple checkmark system. Watching those streaks grow lights a fire under your habit-building hustle.\n\nRemember, habits don’t get rad overnight—it’s about patience AND persistence. Most crush-worthy habits lock in after around 66 days. So, do your thing every morning and soon you’re rolling like a well-oiled productivity machine.\n\n🚀 Morning routine blueprint:\n- Wake consistent\n- Drink water\n- 5-10 min stretch/move\n- Healthy breakfast\n- Mindfulness/journaling\n- Review goals/priorities\n\nBoom. Habit formed, wellness elevated, productivity unlocked. Morning mastery achieved! 🌅🔥	None	building healthy morning routines for wellness and productivity involves understanding habit formation by starting small, attaching new habits to existing ones, and tracking progress. A good morning routine can include waking at a consistent time, drinking water, stretching, having a healthy breakfast, practicing mindfulness or journaling, and reviewing goals. Avoid habits like checking your phone immediately or rushing without preparation. Combining these strategies helps create effective routines that boost energy and mood throughout the day.	True	True	True	6.051645	3649c71d-c2fa-47bb-911c-9790cf6b8f35	019c67e8-743d-75b2-a702-e4ea6e4fc867
5	H0w can I plaan my meels to make shure I get enuff nutriants and stay hydrayed, espeshully when I have a busy sked and need to keep my diet balenced with good water intake and signs of dehydraytion in mind?	Yo, here’s the ultimate playbook to crush meal planning and hydration even when your schedule’s jam-packed:\n\n1. **Scope your week like a boss** – Pinpoint those busy days upfront so you know when you need meals that double as leftovers. Think 3-4 main dishes that fuel you beyond one meal (leftovers = time + energy saved).\n\n2. **Build your nutrient arsenal** – Pack your dishes with complex carbs (whole grains, fruits, veggies), lean proteins (chicken, fish, legumes), and healthy fats (avocado, nuts, olive oil). This combo keeps your body fueled, muscles flexing, and hormones on point.\n\n3. **Snack smart to dodge cravings** – Keep healthy snacks like apples with almond butter or Greek yogurt with honey ready. This strategy crushes impulse eating and keeps your energy stable.\n\n4. **Rock a detailed shopping list** organized by store sections for swift runs and to snag everything green, colorful, and whole – real treasure troves of vitamins and minerals.\n\n5. **Prep ingredients ahead** – Chop, marinate, or cook basics before the chaos hits. When you’re busy, a little prep magic keeps balanced eating effortless.\n\n6. **Hydration game strong** – Slam down about 8 glasses (64 ounces) daily, but tune in to your body’s vibe (activity, heat, size). Start your grind with a glass of water, sling a reusable bottle everywhere, and set hydration alarms on your phone. Bonus points for munching on water-rich foods like cucumbers and watermelon.\n\n7. **Watch for dehydration signals** like dark yellow urine, fatigue, headaches, dry mouth/skin, or dizziness. These alerts are your body screaming for some liquid love.\n\nStick to this flow, and you’ll keep your nutrients stacking and hydration locked down, no sweat—even with that hectic beast of a schedule. Boom, balanced diet + hydration = unstoppable you.	None	To plaan your meels for optimal nutriants and stay hydrayed, start by reviewing your weekly sked and choosing main dishez that can be prepped in advans. Incorporate a balenced diet with complex carbs, lean proteens, and healthy fats, and include water-rich foods like cucumbers and watermelons. Remember to drink at least 8 glasses of water daily, and watch for signs of dehydraytion such as dark urine or headaches. Using a detailed shopping list and prepping ingredients ahead can help you stick to your plan, even on busy days, ensuring you stay nourished and hydrayed.	True	True	True	6.515720	27d3ecd1-7b75-4f78-9940-cefc5f568d83	019c67e8-c8bc-7060-b16a-5492a8fd1335
6	How can stress reduction be achieved through deep breathing and muscle relaxation in daily practice?	Yo, here’s how you crush stress with deep breathing and muscle relaxation like a total boss:\n\nWith **deep breathing**, you’re hacking your nervous system for instant chill. The magic rhythm is inhale for 4 counts, hold for 4, then exhale for 4. This 4-4-4 pattern floods your brain with oxygen, dials down your heart rate, and flips your stress switch to OFF. Think of it as giving your brain a mini-vacation anytime, anywhere.\n\nThen, **progressive muscle relaxation** is like giving your body a reset button. You tense a muscle group—starting from your toes, moving all the way up to your head—hold that tension, then release it with surgical precision. Each release sends a wave of calm, erasing built-up tension like a storm clearing the skies.\n\nIntegrate these into daily life and stress doesn’t stand a chance. They’re your Swiss Army knives for instant cool vibes and long-term zen, perfect for rebooting your mind and body whenever chaos tries to crash your party. Stay rad, stay relaxed.	None	Stress reduction can be achieved through daily practices such as deep breathing, muscle relaxation, grounding, walking, and listening to calming music, which help promote relaxation and emotional well-being.	True	True	True	4.611912	9412076c-3bc7-4d1c-9d17-960fb331ad62	019c67e9-1e84-7062-8bde-753c7fc43128
7	Factors influence mental health and sleep stages how they connect?	Alright, here’s the sick lowdown on how factors influence mental health *and* tie into the wild ride of sleep stages, based on the dopest context you've dropped:\n\n**Mental health** isn’t playing solo—it’s a symphony of variables: biology (genetics, brain chemistry, hormones), life experiences (trauma, abuse), family history, social environment, physical health (hello, chronic illness and sleep quality), and lifestyle habits (exercise, substance use, stress management). Each factor can pump or drain your mental health vibes.\n\nNow, **sleep stages**—especially REM sleep—are like the emotional DJ remixing your day’s emotional memories, helping keep your brain’s emotional playlist balanced. When those factors mess with your mental health, they can throw off your sleep architecture (think insomnia or hypersomnia), which then feeds back to worsen mental health. \n\nHere’s the dazzling connection: **poor sleep screws with emotional regulation, stress coping, and boosts risk for depression and anxiety**, while mental health issues often wreck your sleep (like nightmares in PTSD or chronic insomnia in anxiety). So, if your mental health factors are out of whack, your sleep stages get thrown off rhythm, which in turn deepens mental health struggles—a loop of epic mischief.\n\nIn short: Factors impact mental health → mental health disturbs sleep stages (like REM) → disrupted sleep sabotages emotional balance → hurting mental health again. It’s a bidirectional beast, so mastering both mental health factors and sleep quality is the game-changer.\n\nThat’s the rad nexus of mental health factors and sleep stages—straight from the brain’s backstage pass!	None	Factors influencing mental health include biological aspects, life experiences, family history, social environment, physical health, and lifestyle choices. Sleep occurs in stages like REM and non-REM, which are crucial for mental well-being. Poor sleep can worsen mental health, and mental health issues can disrupt sleep stages, showing their connection.	True	True	True	5.073060	b0390ab6-043a-4e80-bbb0-356e7abb932a	019c67e9-6780-75c3-a0b7-8ee1652e90d7
8	How do Licensed Clinical Social Workers help with setting boundaries and managing digital mental health issues?	Yo, let me drop some knowledge bombs straight from the guide! Licensed Clinical Social Workers (LCSWs) are like boundary-setting ninjas and digital mental health maestros. They help you **pinpoint where your limits lie and what vibes throw you off**, guiding you to **communicate your boundaries with clear, direct “I” statements**—making sure you hold your ground even when pushback is real. They’re there to **coach you through practicing self-compassion when boundary-setting gets tough**, because they know it’s not just about saying no—it’s about protecting your mental hustle and ensuring healthy connections all around.\n\nWhen it comes to managing digital mental health, these pros help you navigate the digital chaos—guiding you to **set intentional social media limits**, **unfollow digital energy drainers**, and **schedule detox periods to unplug and recharge**. They encourage mindful tech use, helping you dodge doom scrolling, FOMO-triggered compulsions, and toxic online comparison traps. Plus, they’ll push you to prioritize **real-life face-to-face interactions** and critically evaluate the digital info avalanche—because your mental health deserves to be curated like a rad playlist, not a chaotic noise fest.\n\nIn sum, LCSWs are your mental health DJs, mixing strong boundary beats with digital self-care anthems so you can thrive IRL and online like the rockstar you are. Boom.	None	Licensed Clinical Social Workers provide therapy and connect clients with resources, which can include guidance on setting and maintaining boundaries, such as identifying limits, communicating clearly, and enforcing them consistently. They also address digital mental health challenges by helping individuals develop strategies like setting time limits for social media, scheduling digital detoxes, and practicing mindful technology use to improve psychological well-being.	True	True	True	2.776680	88ba6507-a111-4cf5-a346-b1187e186912	019c67e9-ae22-7921-8577-5bec4d87e598
9	How does exercise help with mental health according to the context?	Yo, exercise is like the ultimate brain booster powering your mental health to legendary status. According to the context, when you get moving, your body drops a cocktail of endorphins and enkephalins — those natural mood elevators that make you feel like a superhero. It also dials down the stress monsters cortisol and adrenaline, so your brain chills out hardcore. Plus, exercise cranks up serotonin and norepinephrine, neurotransmitters that light up your mood and sharpen your mind.\n\nBut wait, it gets radder: exercise sparks neurogenesis, meaning your brain grows fresh new cells—like leveling up your mental game. It also boosts sleep quality, gives you that sweet sense of accomplishment, and connects you socially if you roll with group workouts or outdoor green exercise vibes.\n\nBottom line? Exercise is a full-spectrum mental health powerhouse—lifting mood, cutting stress, rewiring your brain, and firing up your mojo. Stay moving, stay winning!	None	Exercise affects the brain in multiple beneficial ways, including increasing neurotransmitter levels, promoting neuroplasticity, and reducing inflammation. It releases endorphins and enkephalins (natural mood elevators), reduces cortisol and adrenaline (stress hormones), increases serotonin and norepinephrine production, promotes growth of new brain cells (neurogenesis), improves sleep quality and duration, provides a sense of accomplishment and self-efficacy, and offers social interaction opportunities.	True	True	True	2.845476	4da5dd8e-aa17-4e14-bbc6-07537ee4157e	019c67e9-ddd9-7772-a7b6-d520748dfcc7
10	What is the significance of the University of Massachusetts Medical Center in the context of mindfulness-based therapies?	Oh, strap in for this nugget of mindfulness magic! The University of Massachusetts Medical Center is the OG birthplace of Mindfulness-Based Stress Reduction (MBSR), the legendary 8-week program masterminded by Jon Kabat-Zinn. This program revolutionized how we blend ancient meditation vibes with cutting-edge therapeutic mojo—teaching everything from body scans to yoga to walking meditation—all engineered to crush stress and amplify your well-being. So yeah, that institution didn't just contribute; it *sparked* a mindfulness movement that’s scientifically backed and globally respected. Total altars of zen, but with a lab-coat twist!	None	The University of Massachusetts Medical Center developed Mindfulness-Based Stress Reduction (MBSR), an 8-week program that teaches mindfulness meditation and yoga to reduce stress and improve well-being.	True	True	True	3.229323	2299a097-2d76-4bb7-af81-40711c32e54a	019c67ea-28b3-7e13-865d-f9ee9b79f7f6
11	As a Holistic Wellness Coach, how can understanding the concept of mental health contribute to guiding individuals toward achieving overall well-being through practical strategies and mindful living?	Yo, as a Holistic Wellness Coach, unlocking the true vibe of mental health is like finding the master key to the wellness kingdom. Mental health isn’t just about dodging illness—it’s this vibrant spectrum where emotional, psychological, and social wellness collide, shaping how peeps think, feel, and move through life’s waves.\n\nUnderstanding that mental health means realizing your own powers, managing everyday stress like a boss, staying productive, and contributing to your crew lets you coach with mad precision. You're not just slapping on bandaids; you’re crafting badass, sustainable strategies that honor the *whole* person.\n\nHere’s the ultra-rad playbook from the context to turbocharge your coaching mojo:\n\n- **Mind-Body Sync:** Mental health and physical health are dance partners. Exercise fires off endorphins, zapping anxiety and depression. Nutrition tunes the gut-brain axis, which controls mood via serotonin superpowers produced in the gut. Coaches can push clean, diverse diets with fiber, fermented foods, and omega-3s to elevate mental vibes naturally.\n\n- **Growth Mindset Magic:** Shifting “I can’t” to “I can’t yet” transforms setbacks into launchpads. Guiding clients to celebrate efforts over just results fortifies resilience and self-compassion—key ingredients for mental toughness.\n\n- **Therapeutic Hacks:** CBT vibes help folks flip negative thought scripts into empowering ones, breaking cycles of emotional distress. Incorporating CBT-inspired tools offers clients pragmatic mental resets.\n\n- **Daily Rituals:** Journaling is a ninja move to unpack tangled emotions, boost self-awareness, and spot patterns in mental health. This expressive writing sidekick even lowers stress and levels up immune function.\n\nBy owning this mental health blueprint, you steer people beyond surviving toward *thriving*—balancing mind, body, and spirit through mindful living and actionable practices that anchor them in strength, clarity, and flow. That’s not just coaching, that’s mental health mastery on full display, lighting up wellness like a cosmic beacon. 🌟	None	Mental health encompasses our emotional, psychological, and social well-being, affecting how we think, feel, and act. It is important at every stage of life and exists on a spectrum, with everyone's experience being unique. Factors contributing to mental health include biological factors, life experiences, family history, social environment, physical health, and lifestyle choices. Good mental health is not merely the absence of mental illness but involves realizing one's abilities, coping with normal stresses, working productively, and contributing to the community. Understanding these aspects allows a Holistic Wellness Coach to guide individuals in adopting lifestyle habits that promote mental well-being, such as managing stress, fostering supportive relationships, and maintaining physical health through mindful practices.	True	True	True	6.110734	8633b70e-ba3d-4866-b065-35be09b7073f	019c67ea-65ed-71e0-a8cf-41f5197cab2b
🏗️ Activity #2: Analyze Evaluation Results
Provide a screenshot of the difference between the two chains in LangSmith, and explain why you believe certain metrics changed in certain ways.

Answer:
Default Chain : https://smith.langchain.com/o/2bf8dd81-d1e2-410c-a798-2636bdfe0958/datasets/e027d2f3-0749-4da0-a81e-35a5047090f8/compare?selectedSessions=eee95471-dd8a-4f6b-a1ac-160731fa05bb

Screen Shots: image.png image.png

Dopeness Chain : https://smith.langchain.com/o/2bf8dd81-d1e2-410c-a798-2636bdfe0958/datasets/e027d2f3-0749-4da0-a81e-35a5047090f8/compare?selectedSessions=390026a4-236b-481e-ba2c-5ee30cc1f92d

Screen Shots: image.png image.png

Comparison link : https://smith.langchain.com/o/2bf8dd81-d1e2-410c-a798-2636bdfe0958/datasets/e027d2f3-0749-4da0-a81e-35a5047090f8/compare?selectedSessions=eee95471-dd8a-4f6b-a1ac-160731fa05bb%2C390026a4-236b-481e-ba2c-5ee30cc1f92d&baseline=eee95471-dd8a-4f6b-a1ac-160731fa05bb

Screen Shots: image.png image.png image.png

Observed Metric Changes
Default Chain
QA avg: 0.917
Helpfulness avg: 0.917
Dopeness avg: 0.417
Latency (P50): 4.22s
Total Tokens: 18,093
Total Cost: $0.0101
Dopeness Chain
QA avg: 1.00
Helpfulness avg: 1.00
Dopeness avg: 1.00
Latency (P50): 4.92s
Total Tokens: 15,625
Total Cost: $0.0109
Feedback Score Changes
i. QA: 0.917 → 1.00 One previously incorrect example became correct.Looking at the baseline run, example #8 returned “I don’t know.” In the experimental run, that query was answered properly.This indicates that the improvement was not stylistic — it was retrieval-related.

Likely causes:

Larger chunk size (better context completeness)
Improved embedding model (better semantic recall)
This confirms that retrieval improvements directly influence factual correctness.
ii. Helpfulness: 0.917 → 1.00 Helpfulness increased alongside QA.Since the only zero case in baseline corresponded to the “I don’t know” response, once retrieval improved and the system produced a grounded answer, helpfulness naturally increased. Additionally, the modified prompt encouraged:

More structured answers
More confident tone
Fuller explanations
This likely improved the evaluator’s perception of usefulness.

iii. Dopeness: 0.417 → 1.00 This is the most dramatic change. Baseline responses were correct but stylistically neutral or generic. The evaluator marked many of them as 0 for dopeness. In the experiment:

The prompt explicitly instructed the model to “crank up the dopeness.”
Responses became energetic, expressive, and stylized.
Because the evaluator measures engagement and non-generic tone, scores jumped to 1.00 across all examples.
This demonstrates: Prompt engineering strongly influences stylistic metrics.

Latency Analysis
P50 latency increased: 4.22s → 4.92s

P99 latency increased significantly (visible in the chart).

Reasons:

Larger chunk size increases context passed to the model.
Embedding model upgrade increases retrieval computation.
Longer, more expressive outputs increase generation time.
The trade-off is clear: Higher quality and engagement → slightly slower responses.

Token Count Changes
Total tokens decreased overall: 18,093 → 15,625

However:

Input tokens decreased (likely due to chunking changes reducing redundant retrieval overlap).
Output tokens increased (2,363 → 3,794), because responses became more verbose and stylized.
So:

Retrieval context became more efficient,
Generation became more expressive.
Cost Changes
Total cost: 
0.0109

Cost increased slightly due to:

Larger output token count
More expressive responses
This reflects a typical RAG trade-off: Better experience → marginally higher generation cost.

What This Experiment Demonstrates
This comparison shows three independent levers affecting RAG performance:

Retrieval Architecture
Chunk size affects context completeness.
Embedding model affects semantic recall.
These primarily impact QA and helpfulness.
ii. Prompt Engineering - Directly influences tone and engagement. - Strongly impacts stylistic metrics like dopeness. - Has secondary effects on helpfulness.

iii. System Trade-offs - Improved quality increases latency. - Increased verbosity increases output tokens and cost.

Final Insight
The key takeaway from this experiment is:

Retrieval improvements drive correctness.
Prompt improvements drive engagement.
Both can be measured independently using LangSmith.

The experiment clearly shows that evaluation metrics respond differently to architectural changes versus prompt changes, and LangSmith’s comparison view makes those trade-offs visible and quantifiable.

Advanced Build : Reproduce the RAGAS Synthetic Data Generation Steps - but utilize a LangGraph Agent Graph, instead of the Knowledge Graph approach. This generation should leverage the Evol Instruct method to generate synthetic data.
Design
Below is Advanced Build implementation plan. It follows the README requirement:

Use a LangGraph Agent Graph (not RAGAS KnowledgeGraph)
Use Evol-Instruct-style evolution to generate synthetic questions
Output final state with:

List[dict] evolved questions + IDs + evolution type
List[dict] answers keyed by question_id
List[dict] contexts keyed by question_id
What is built? A LangGraph pipeline that takes LangChain docs → generates seed questions → evolves them (simple / multi-context / reasoning) → retrieves supporting contexts → answers using “context-only” rule → returns the 3 required output lists.

Minimal architecture (LangGraph nodes)

State contains

docs: List[Document]
seed_questions: List[dict] (id, question)
evolved_questions: List[dict] (id, parent_id, evolution_type, question)
contexts: List[dict] (question_id, contexts)
answers: List[dict] (question_id, answer)
Nodes

build_index → creates vectorstore + retriever (once)
generate_seed_questions → creates initial questions from docs
evolve_questions → Evol-Instruct evolution into 3 types
retrieve_contexts → gets top-k chunks for each evolved question
answer_questions → answers using ONLY retrieved context (“I don’t know” otherwise)
finalize_outputs → formats exactly as required
Step 1 : Imports + LLM + embeddings + splitter + vector store
from typing import TypedDict, List, Dict, Any, Optional
from uuid import uuid4

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore

from langgraph.graph import StateGraph, END

# Models (keep them consistent with your notebook)
gen_llm = ChatOpenAI(model="gpt-4.1-mini")   # question generation + evolution
ans_llm = ChatOpenAI(model="gpt-4.1-mini")   # answering
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
Step 2 : Define Evol-Instruct evolution prompts (Simple / Multi-context / Reasoning)
This is the “Evol-Instruct” part: we are applying controlled transformations to increase complexity.

EVOL_SIMPLE_PROMPT = """You are evolving an instruction (question).
Rewrite the question to be slightly more specific and constrained, without changing the topic.
Add 1-2 constraints. Keep it answerable from the same domain.

Return ONLY the evolved question text.

Original question:
{question}
"""

EVOL_MULTI_CONTEXT_PROMPT = """You are evolving an instruction (question) to require multi-context evidence.
Rewrite the question so that answering it requires combining information from at least TWO different areas/topics in the corpus.
Make it explicit that both aspects must be covered.

Return ONLY the evolved question text.

Original question:
{question}
"""

EVOL_REASONING_PROMPT = """You are evolving an instruction (question) to require reasoning steps.
Rewrite the question so that the answer must explain a cause-effect relationship, comparison, or a step-by-step reasoning chain.
Do NOT ask for opinions; keep it grounded and answerable from documents.

Return ONLY the evolved question text.

Original question:
{question}
"""
Step 3 : Define the LangGraph State
class SDGState(TypedDict):
    docs: List[Document]

    # built artifacts
    retriever: Any

    # generation artifacts
    seed_questions: List[Dict[str, Any]]
    evolved_questions: List[Dict[str, Any]]

    # evidence + outputs
    contexts: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]
Step 4 : Node 1: Build index (vectorstore + retriever)
def build_index(state: SDGState) -> SDGState:
    docs = state["docs"]
    chunks = text_splitter.split_documents(docs)

    vs = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name=f"adv_sdg_{uuid4().hex[:8]}"
    )
    retriever = vs.as_retriever(search_kwargs={"k": 6})

    state["retriever"] = retriever
    state["seed_questions"] = []
    state["evolved_questions"] = []
    state["contexts"] = []
    state["answers"] = []
    return state
Step 5 : Node 2: Generate seed questions (small set)
SEED_Q_PROMPT = """You are generating evaluation questions for a RAG system.
Given the corpus summary below, generate {n} user questions that are answerable from this corpus.
Questions should be realistic and not trivial.

Return as a JSON list of strings only.

Corpus summary:
{summary}
"""

def generate_seed_questions(state: SDGState, n: int = 6) -> SDGState:
    # quick corpus summary (lightweight)
    corpus_text = "\n\n".join([d.page_content[:1200] for d in state["docs"]])
    summary = gen_llm.invoke(f"Summarize key topics in this corpus in 12-15 bullets:\n\n{corpus_text}").content

    raw = gen_llm.invoke(SEED_Q_PROMPT.format(summary=summary, n=n)).content

    # robust parse: accept either JSON list or fallback split-lines
    import json
    questions = []
    try:
        questions = json.loads(raw)
        if not isinstance(questions, list):
            questions = []
    except Exception:
        # fallback heuristic
        questions = [q.strip("- ").strip() for q in raw.split("\n") if len(q.strip()) > 10]

    seed = []
    for q in questions[:n]:
        seed.append({"id": f"seed_{uuid4().hex[:8]}", "question": q})

    state["seed_questions"] = seed
    return state
Step 6 : Node 3: Evolve questions (3 evolution types per seed)
This satisfies “Graph should handle: simple evolution, multi-context evolution, reasoning evolution”.

def _evolve_one(question: str, prompt: str) -> str:
    out = gen_llm.invoke(prompt.format(question=question)).content.strip()
    # basic cleanup
    return out.replace('"', "").strip()

def evolve_questions(state: SDGState) -> SDGState:
    evolved = []

    for item in state["seed_questions"]:
        parent_id = item["id"]
        q = item["question"]

        simple_q = _evolve_one(q, EVOL_SIMPLE_PROMPT)
        evolved.append({
            "id": f"evo_{uuid4().hex[:8]}",
            "parent_id": parent_id,
            "evolution_type": "simple",
            "question": simple_q
        })

        mc_q = _evolve_one(q, EVOL_MULTI_CONTEXT_PROMPT)
        evolved.append({
            "id": f"evo_{uuid4().hex[:8]}",
            "parent_id": parent_id,
            "evolution_type": "multi_context",
            "question": mc_q
        })

        reason_q = _evolve_one(q, EVOL_REASONING_PROMPT)
        evolved.append({
            "id": f"evo_{uuid4().hex[:8]}",
            "parent_id": parent_id,
            "evolution_type": "reasoning",
            "question": reason_q
        })

    state["evolved_questions"] = evolved
    return state
Step 7 : Node 4: Retrieve contexts for each evolved question
def retrieve_contexts(state: SDGState) -> SDGState:
    retriever = state["retriever"]
    contexts = []

    for q in state["evolved_questions"]:
        docs = retriever.invoke(q["question"])
        ctxs = []
        for d in docs:
            # keep small payload; still readable
            ctxs.append({
                "source": d.metadata.get("source", ""),
                "text": d.page_content
            })
        contexts.append({"question_id": q["id"], "contexts": ctxs})

    state["contexts"] = contexts
    return state
Step 8 : Node 5: Answer strictly from context (“I don’t know”)
ANSWER_PROMPT = """Answer the question using ONLY the provided context.
If the answer is not supported by the context, reply exactly: "I don't know".

Question:
{question}

Context:
{context}
"""

def answer_questions(state: SDGState) -> SDGState:
    # map question_id -> contexts
    ctx_map = {c["question_id"]: c["contexts"] for c in state["contexts"]}
    answers = []

    for q in state["evolved_questions"]:
        ctxs = ctx_map.get(q["id"], [])
        # join a few chunks
        joined = "\n\n---\n\n".join([c["text"][:1800] for c in ctxs[:6]])

        pred = ans_llm.invoke(
            ANSWER_PROMPT.format(question=q["question"], context=joined)
        ).content.strip()

        answers.append({"question_id": q["id"], "answer": pred})

    state["answers"] = answers
    return state
Step 9 : Node 6: Finalize outputs (exact required lists)
def finalize_outputs(state: SDGState) -> SDGState:
    # The README wants (at least):
    # 1) List(dict): Evolved Questions, IDs, Evolution Type
    # 2) List(dict): Question IDs + Answer
    # 3) List(dict): Question IDs + relevant Context(s)

    # We already have them in state; just ensure final shape is clean.
    state["evolved_questions"] = [
        {"question_id": q["id"], "parent_id": q["parent_id"], "evolution_type": q["evolution_type"], "question": q["question"]}
        for q in state["evolved_questions"]
    ]
    state["contexts"] = [
        {"question_id": c["question_id"], "contexts": c["contexts"]}
        for c in state["contexts"]
    ]
    state["answers"] = [
        {"question_id": a["question_id"], "answer": a["answer"]}
        for a in state["answers"]
    ]
    return state
Step 10 : Build + run the LangGraph
graph = StateGraph(SDGState)

graph.add_node("build_index", build_index)
graph.add_node("generate_seed_questions", generate_seed_questions)
graph.add_node("evolve_questions", evolve_questions)
graph.add_node("retrieve_contexts", retrieve_contexts)
graph.add_node("answer_questions", answer_questions)
graph.add_node("finalize_outputs", finalize_outputs)

graph.set_entry_point("build_index")

graph.add_edge("build_index", "generate_seed_questions")
graph.add_edge("generate_seed_questions", "evolve_questions")
graph.add_edge("evolve_questions", "retrieve_contexts")
graph.add_edge("retrieve_contexts", "answer_questions")
graph.add_edge("answer_questions", "finalize_outputs")
graph.add_edge("finalize_outputs", END)

app = graph.compile()


# Run it: input is your list of LangChain documents `docs`
final_state = app.invoke({"docs": docs})

evolved_questions = final_state["evolved_questions"]
answers = final_state["answers"]
contexts = final_state["contexts"]

print("Evolved questions:", len(evolved_questions))
print("Answers:", len(answers))
print("Contexts:", len(contexts))

# Show a sample
evolved_questions[:3], answers[:2], contexts[:1]
Evolved questions: 18
Answers: 18
Contexts: 18
([{'question_id': 'evo_10adf1c7',
   'parent_id': 'seed_c00d8a73',
   'evolution_type': 'simple',
   'question': "According to the handbook, how do specific biological factors such as genetics and neurochemistry influence an individual's mental health?"},
  {'question_id': 'evo_72f9d40c',
   'parent_id': 'seed_c00d8a73',
   'evolution_type': 'multi_context',
   'question': "How do biological factors influence an individual's mental health according to the handbook, and how do environmental or social factors interact with these biological influences to shape overall mental well-being?"},
  {'question_id': 'evo_c3f09f31',
   'parent_id': 'seed_c00d8a73',
   'evolution_type': 'reasoning',
   'question': "According to the handbook, what are the biological factors that influence an individual's mental health, and how do these factors interact to affect mental health outcomes?"}],
 [{'question_id': 'evo_10adf1c7',
   'answer': "According to the handbook, specific biological factors such as genetics, brain chemistry, and hormones contribute to an individual's mental health. These biological factors play a role alongside other influences like life experiences, family history, and social environment in shaping mental well-being."},
  {'question_id': 'evo_72f9d40c',
   'answer': "According to the handbook, biological factors such as genetics, brain chemistry, and hormones influence an individual's mental health by contributing to their overall mental well-being and risk for mental health conditions. Environmental and social factors, such as life experiences (trauma, abuse, significant life changes), family history, social environment (relationships, community, support systems), physical health, and lifestyle choices, interact with these biological influences to shape mental health outcomes. For example, chronic stress (an environmental factor) increases cortisol levels (a biological factor), which weakens the immune system and affects mental health. Moreover, social connections are essential for mental health, helping to mitigate risks such as loneliness, which is a significant public health concern. In sum, mental health results from a complex interplay between biological predispositions and environmental/social experiences that together influence an individual's mental well-being."}],
 [{'question_id': 'evo_10adf1c7',
   'contexts': [{'source': 'data/MentalHealthGuide.txt',
     'text': 'Factors that contribute to mental health include:\n- Biological factors: Genetics, brain chemistry, and hormones\n- Life experiences: Trauma, abuse, or significant life changes\n- Family history: Mental health conditions in close relatives\n- Social environment: Relationships, community, and support systems\n- Physical health: Chronic illness, sleep quality, and nutrition\n- Lifestyle choices: Exercise habits, substance use, and stress management\n\nChapter 2: Common Mental Health Conditions\n\nAnxiety Disorders\nAnxiety disorders are the most common mental health conditions, affecting approximately 40 million adults in the United States. They involve persistent, excessive worry that interferes with daily activities.'},
    {'source': 'data/MentalHealthGuide.txt',
     'text': "The Mental Health and Psychology Handbook\nA Practical Guide to Understanding and Improving Mental Well-being\n\nPART 1: FOUNDATIONS OF MENTAL HEALTH\n\nChapter 1: What Is Mental Health?\n\nMental health encompasses our emotional, psychological, and social well-being. It affects how we think, feel, and act. Mental health is important at every stage of life, from childhood and adolescence through adulthood. According to the World Health Organization, mental health is a state of well-being in which an individual realizes their own abilities, can cope with the normal stresses of life, can work productively, and is able to make a contribution to their community.\n\nGood mental health is not simply the absence of mental illness. A person can experience mental health challenges without having a diagnosable condition, and someone with a diagnosed condition can still experience periods of good mental health. Mental health exists on a spectrum, and everyone's experience is unique."},
    {'source': 'data/MentalHealthGuide.txt',
     'text': 'Chapter 3: The Mind-Body Connection\n\nResearch consistently demonstrates the powerful connection between mental and physical health. Mental health conditions can increase the risk of physical health problems, and physical health issues can affect mental well-being.\n\nHow mental health affects physical health:\n- Chronic stress increases cortisol levels, weakening the immune system\n- Depression is associated with a 40% higher risk of cardiovascular disease\n- Anxiety can lead to chronic muscle tension, headaches, and digestive problems\n- Poor mental health often leads to disrupted sleep patterns\n- Mental health conditions may reduce motivation for exercise and healthy eating'},
    {'source': 'data/MentalHealthGuide.txt',
     'text': "How physical health affects mental health:\n- Regular exercise releases endorphins and reduces symptoms of depression and anxiety\n- Poor nutrition can worsen mood and cognitive function\n- Chronic pain is strongly associated with depression\n- Sleep deprivation impairs emotional regulation and increases anxiety\n- Gut health influences mood through the gut-brain axis\n\nThe gut-brain connection is an area of growing research interest. The gut microbiome produces approximately 90% of the body's serotonin, a neurotransmitter crucial for mood regulation. Eating a diverse diet rich in fiber, fermented foods, and omega-3 fatty acids supports both gut health and mental well-being.\n\nPART 2: THERAPEUTIC APPROACHES\n\nChapter 4: Cognitive Behavioral Therapy (CBT)\n\nCognitive Behavioral Therapy is one of the most widely studied and effective forms of psychotherapy. It focuses on identifying and changing negative thought patterns and behaviors that contribute to emotional distress."},
    {'source': 'data/MentalHealthGuide.txt',
     'text': 'Dietary patterns and mental health:\n- Mediterranean diet is associated with 30% lower risk of depression\n- Western diet (high processed food, sugar) is linked to higher depression rates\n- Anti-inflammatory diets may reduce symptoms of depression and anxiety\n- Regular meal timing supports stable mood and energy levels\n- Excessive caffeine can worsen anxiety symptoms\n- Alcohol, while sometimes used to cope, worsens depression and anxiety over time\n\nPART 5: SOCIAL AND ENVIRONMENTAL FACTORS\n\nChapter 14: The Importance of Social Connection\n\nHuman beings are inherently social creatures. Strong social connections are essential for mental health, and loneliness is recognized as a significant public health concern. Research has shown that loneliness increases the risk of premature death by 26%, which is comparable to smoking 15 cigarettes per day.'},
    {'source': 'data/MentalHealthGuide.txt',
     'text': 'Chapter 12: Sleep and Mental Health\n\nSleep and mental health have a bidirectional relationship. Poor sleep can trigger or worsen mental health conditions, while mental health problems often disrupt sleep. Addressing sleep issues is frequently a critical component of mental health treatment.\n\nHow sleep affects mental health:\n- Sleep deprivation impairs emotional regulation and increases reactivity\n- Insufficient sleep raises the risk of developing depression by 2-3 times\n- REM sleep helps process emotional memories and experiences\n- Chronic insomnia is both a symptom and risk factor for anxiety disorders\n- Sleep loss reduces the ability to cope with stress\n\nSleep disorders associated with mental health:\n- Insomnia: Most common sleep complaint in mental health conditions\n- Hypersomnia: Excessive sleeping, often associated with depression\n- Nightmares: Frequently occur with PTSD and anxiety\n- Sleep apnea: Associated with depression and cognitive impairment'}]}])
Summary
In this session, we:

Generated synthetic test data using Ragas' knowledge graph-based approach
Explored query synthesizers for creating diverse question types
Loaded synthetic data into a LangSmith dataset for evaluation
Built and evaluated a RAG chain using LangSmith evaluators
Iterated on the pipeline by modifying chunk size, embedding model, and prompt — then measured the impact
Key Takeaways:
Synthetic data generation is critical for early iteration — it provides high-quality signal without manually creating test data
LangSmith evaluators enable systematic comparison of pipeline versions
Small changes matter — chunk size, embedding model, and prompt modifications can significantly affect evaluation scores