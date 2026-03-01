
Session 11: Advanced Retrieval with LangChain
Learning Objectives:
Understand and implement multiple retrieval strategies for RAG
Compare naive, BM25, multi-query, parent-document, contextual compression, ensemble, and semantic chunking approaches
Build RAG chains over a health and wellness knowledge base using LangChain and QDrant
In the following notebook, we'll explore various methods of advanced retrieval using LangChain!

We'll touch on:

Naive Retrieval
Best-Matching 25 (BM25)
Multi-Query Retrieval
Parent-Document Retrieval
Contextual Compression (a.k.a. Rerank)
Ensemble Retrieval
Semantic chunking
We'll also discuss how these methods impact performance on our set of documents with a simple RAG chain.

There will be two breakout rooms:

🤝 Breakout Room Part #1
Task 1: Getting Dependencies!
Task 2: Data Collection and Preparation
Task 3: Setting Up QDrant!
Task 4-10: Retrieval Strategies
🤝 Breakout Room Part #2
Activity: Evaluate with Ragas
🤝 Breakout Room Part #1
Task 1: Getting Dependencies!
We're going to need a few specific LangChain community packages, like OpenAI (for our LLM and Embedding Model) and Cohere (for our Reranker).

We'll also provide our OpenAI key, as well as our Cohere API key.

NOTE: Create a .env file in this directory with OPENAI_API_KEY and COHERE_API_KEY to avoid being prompted each time.

import os
import getpass
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key:")
if not os.environ.get("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = getpass.getpass("Cohere API Key:")
Task 2: Data Collection and Preparation
We'll be using our Health and Wellness Guide - a comprehensive resource covering exercise, nutrition, sleep, stress management, habits, and common health concerns.

Data Preparation
We'll load the wellness guide as a single document, then split it into smaller chunks using a RecursiveCharacterTextSplitter for our vector store. We also keep the raw (unsplit) document for use with the Parent Document Retriever and Semantic Chunker later.

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("data/HealthWellnessGuide.txt")
raw_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
wellness_docs = text_splitter.split_documents(raw_docs)
Let's verify our data was loaded and split correctly!

print(f"Raw documents: {len(raw_docs)}")
print(f"Split chunks: {len(wellness_docs)}")
print(f"\nExample chunk:\n{wellness_docs[0]}")
Raw documents: 1
Split chunks: 45

Example chunk:
page_content='The Personal Wellness Guide
A Comprehensive Resource for Health and Well-being

PART 1: EXERCISE AND MOVEMENT

Chapter 1: Understanding Exercise Basics

Exercise is one of the most important things you can do for your health. Regular physical activity can improve your brain health, help manage weight, reduce the risk of disease, strengthen bones and muscles, and improve your ability to do everyday activities.' metadata={'source': 'data/HealthWellnessGuide.txt'}
Task 3: Setting up QDrant!
Now that we have our documents, let's create a QDrant VectorStore with the collection name "wellness_guide".

We'll leverage OpenAI's text-embedding-3-small because it's a very powerful (and low-cost) embedding model.

NOTE: We'll be creating additional vectorstores where necessary, but this pattern is still extremely useful.

from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = QdrantVectorStore.from_documents(
    wellness_docs,
    embeddings,
    location=":memory:",
    collection_name="wellness_guide",
)
Task 4: Naive RAG Chain
Since we're focusing on the "R" in RAG today - we'll create our Retriever first.

R - Retrieval
This naive retriever will simply look at each review as a document, and use cosine-similarity to fetch the 10 most relevant documents.

NOTE: We're choosing 10 as our k here to provide enough documents for our reranking process later

naive_retriever = vectorstore.as_retriever(search_kwargs={"k" : 10})
A - Augmented
We're going to go with a standard prompt for our simple RAG chain today! Nothing fancy here, we want this to mostly be about the Retrieval process.

from langchain_core.prompts import ChatPromptTemplate

RAG_TEMPLATE = """\
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
G - Generation
We're going to leverage gpt-4.1-nano as our LLM today, as - again - we want this to largely be about the Retrieval process.

from langchain_openai import ChatOpenAI

chat_model = ChatOpenAI(model="gpt-4.1-nano")
LCEL RAG Chain
We're going to use LCEL to construct our chain.

NOTE: This chain will be exactly the same across the various examples with the exception of our Retriever!

from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser

naive_retrieval_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | naive_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
Let's see how this simple chain does on a few different prompts.

NOTE: You might think that we've cherry picked prompts that showcase the individual skill of each of the retrieval strategies - you'd be correct!

naive_retrieval_chain.invoke({"question" : "What exercises can help with lower back pain?"})["response"].content
'Exercises that can help with lower back pain include:\n\n- **Cat-Cow Stretch:** Start on your hands and knees, alternate arching your back up (cat) and letting it sag down (cow). Perform 10-15 repetitions.\n- **Bird Dog:** From hands and knees, extend opposite arm and leg while keeping your core engaged. Hold each for 5 seconds and switch sides. Do 10 repetitions per side.\n- **Pelvic Tilts:** Lie on your back with knees bent, flatten your back against the floor by tightening your abs and tilting your pelvis slightly upward. Hold for 10 seconds and repeat 8-12 times.\n\nThese gentle exercises can help alleviate lower back discomfort and prevent future episodes.'
naive_retrieval_chain.invoke({"question" : "How does sleep affect overall health?"})["response"].content
'Sleep has a significant impact on overall health. Adequate sleep—typically 7 to 9 hours for adults—supports physical repair by enabling the body to recover tissues and regenerate. It also plays a crucial role in mental well-being, aiding in memory consolidation, learning, and emotional regulation. During sleep, the body releases hormones that regulate growth and appetite, contributing to a healthy physique and metabolic balance. Poor sleep or sleep disorders like insomnia can negatively affect the immune system, increase stress levels, and impair cognitive functions. Therefore, maintaining good sleep hygiene and creating an optimal sleep environment are essential for overall health and well-being.'
naive_retrieval_chain.invoke({"question" : "What are some natural remedies for stress and headaches?"})["response"].content
'Some natural remedies for stress and headaches include:\n\n- For headaches:\n  - Drink water and stay hydrated\n  - Apply a cold or warm compress to the head or neck\n  - Rest in a dark, quiet room\n  - Gentle massage of temples and neck\n  - Use peppermint or lavender essential oils\n  - Maintain a regular sleep schedule\n  - Be mindful of triggers such as dehydration, stress, poor sleep, skipped meals, eye strain, weather changes, and certain foods\n\n- For stress relief:\n  - Practice deep breathing exercises, such as inhaling for 4 counts, holding, and exhaling\n  - Engage in progressive muscle relaxation, tensing and releasing muscle groups\n  - Use grounding techniques by identifying things you see, hear, feel, smell, and taste\n  - Take short walks, especially in nature\n  - Listen to calming music\n\nThese approaches, along with maintaining good hydration, adequate sleep, and managing stress through mindfulness and relaxation techniques, can help alleviate stress and headache symptoms naturally.'
Overall, this is not bad! Let's see if we can make it better!

Task 5: Best-Matching 25 (BM25) Retriever
Taking a step back in time - BM25 is based on Bag-Of-Words which is a sparse representation of text.

In essence, it's a way to compare how similar two pieces of text are based on the words they both contain.

This retriever is very straightforward to set-up! Let's see it happen down below!

import sys
!{sys.executable} -m pip install -U rank_bm25
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(wellness_docs)
We'll construct the same chain - only changing the retriever.

bm25_retrieval_chain = (
    {"context": itemgetter("question") | bm25_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
Let's look at the responses!

bm25_retrieval_chain.invoke({"question" : "What exercises can help with lower back pain?"})["response"].content
'Exercises that can help with lower back pain include the Cat-Cow Stretch, Bird Dog, and Pelvic Tilts. The Cat-Cow Stretch involves starting on hands and knees, then arching your back up (cat) and letting it sag down (cow), performing 10-15 repetitions. The Bird Dog involves extending opposite arm and leg while on hands and knees, holding for around 5 seconds, and doing about 10 repetitions per side. Pelvic Tilts are done lying on your back with knees bent, tightening your abs and tilting your pelvis upward to flatten your back against the floor, holding for 10 seconds and repeating 8-12 times.'
bm25_retrieval_chain.invoke({"question" : "How does sleep affect overall health?"})["response"].content
'Sleep significantly affects overall health by promoting restorative bodily functions and mental well-being. Maintaining a consistent sleep schedule and creating an optimal sleep environment—such as keeping the bedroom cool, dark, and quiet—helps ensure quality sleep. Good sleep hygiene practices, including limiting screen exposure before bed, avoiding caffeine late in the day, and relaxing routines, support healthy sleep patterns. Adequate sleep contributes to a stronger immune system, improved mood, better cognitive function, and overall physical health. Conversely, poor sleep or insomnia can negatively impact mental and physical health, highlighting the importance of sleep for overall wellness.'
bm25_retrieval_chain.invoke({"question" : "What are some natural remedies for stress and headaches?"})["response"].content
"Some natural remedies for stress and headaches include relaxation techniques such as deep breathing exercises and progressive muscle relaxation, herbal teas like chamomile or valerian root, and practices like meditation. Additionally, staying well-hydrated, ensuring adequate sleep, and managing stress through mindfulness can help reduce headaches and stress-related symptoms. However, it's always best to consult with a healthcare professional before starting any new remedies."
It's not clear that this is better or worse, if only we had a way to test this (SPOILERS: We do, the second half of the notebook will cover this)

❓ Question #1:
Give an example query where BM25 is better than embeddings and justify your answer.

Answer:
An example query where BM25 would outperform embeddings is:

“What is the recommended dosage of ibuprofen 800mg for acute lower back pain?”

BM25 performs better in this case because the query contains highly specific lexical terms such as “ibuprofen 800mg” and “acute lower back pain.” BM25 is a term-matching algorithm that relies on exact keyword overlap and term frequency. If the document explicitly contains “ibuprofen 800mg,” BM25 will rank it highly because of direct token matching and inverse document frequency weighting.

Embedding-based retrieval, on the other hand, focuses on semantic similarity rather than exact token matches. It may retrieve documents discussing “pain relief medication” or “NSAIDs for back pain” without specifically mentioning “800mg dosage.” While semantically related, those results might miss the precise dosage detail required by the query.

BM25 is particularly strong when: - The query contains exact product names, codes, or dosages - Specific terminology matters - Precision is more important than semantic flexibility

In contrast, embeddings are stronger when: - The query is conceptual or paraphrased - The wording differs from the document - Semantic meaning matters more than exact wording

In this example, because the user is asking for a very specific dosage tied to an exact medication strength, lexical precision matters more than semantic generalization, making BM25 the better retrieval method.

Task 6: Contextual Compression (Using Reranking)
Contextual Compression is a fairly straightforward idea: We want to "compress" our retrieved context into just the most useful bits.

There are a few ways we can achieve this - but we're going to look at a specific example called reranking.

The basic idea here is this:

We retrieve lots of documents that are very likely related to our query vector
We "compress" those documents into a smaller set of more related documents using a reranking algorithm.
We'll be leveraging Cohere's Rerank model for our reranker today!

All we need to do is the following:

Create a basic retriever
Create a compressor (reranker, in this case)
That's it!

Let's see it in the code below!

#from langchain_community.retrievers import ContextualCompressionRetriever
#from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
#from langchain.retrievers import ContextualCompressionRetriever

# import sys
# !{sys.executable} -m pip install -U langchain-classic

# import sys
# !{sys.executable} -m pip install -U langchain-cohere cohere


from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


compressor = CohereRerank(model="rerank-v3.5")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=naive_retriever
)
Let's create our chain again, and see how this does!

contextual_compression_retrieval_chain = (
    {"context": itemgetter("question") | compression_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
contextual_compression_retrieval_chain.invoke({"question" : "What exercises can help with lower back pain?"})["response"].content
'Exercises that can help with lower back pain include:\n\n- **Cat-Cow Stretch:** Start on your hands and knees, then alternate between arching your back up (cat) and letting it sag down (cow). Do 10-15 repetitions.\n\n- **Bird Dog:** From a hands-and-knees position, extend your opposite arm and leg while keeping your core engaged. Hold for 5 seconds, then switch sides. Aim for 10 repetitions per side.\n\n- **Pelvic Tilts:** Lie on your back with knees bent, flatten your back against the floor by tightening your abs and tilting your pelvis slightly. Hold for 10 seconds and repeat 8-12 times.\n\nThese gentle exercises can help alleviate discomfort and prevent future episodes of lower back pain.'
contextual_compression_retrieval_chain.invoke({"question" : "How does sleep affect overall health?"})["response"].content
'Sleep has a significant impact on overall health. It is essential for the repair of tissues, regulation of growth hormones, and maintenance of cognitive functions like memory and learning. During sleep, the body undergoes cycles that include both REM and non-REM stages, each playing a vital role in physical and mental health. Adequate sleep—generally 7-9 hours per night—supports mental well-being, boosts immune function, and helps regulate appetite. Creating a proper sleep environment, such as maintaining an optimal temperature, darkness, quietness, and comfort, can improve sleep quality. Poor sleep or disorders like insomnia can negatively affect overall health, highlighting the importance of good sleep habits.'
contextual_compression_retrieval_chain.invoke({"question" : "What are some natural remedies for stress and headaches?"})["response"].content
'Some natural remedies for stress and headaches include practicing deep breathing, progressive muscle relaxation, grounding techniques, taking short walks in nature, listening to calming music, staying hydrated by drinking water, applying cold or warm compresses to the head or neck, resting in a dark, quiet room, gently massaging the temples and neck, using peppermint or lavender essential oils, and maintaining a regular sleep schedule.'
We'll need to rely on something like Ragas to help us get a better sense of how this is performing overall - but it "feels" better!

Task 7: Multi-Query Retriever
Typically in RAG we have a single query - the one provided by the user.

What if we had....more than one query!

In essence, a Multi-Query Retriever works by:

Taking the original user query and creating n number of new user queries using an LLM.
Retrieving documents for each query.
Using all unique retrieved documents as context
So, how is it to set-up? Not bad! Let's see it down below!

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
#from langchain.retrievers.multi_query import MultiQueryRetriever


multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=naive_retriever, llm=chat_model
) 
multi_query_retrieval_chain = (
    {"context": itemgetter("question") | multi_query_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
multi_query_retrieval_chain.invoke({"question" : "What exercises can help with lower back pain?"})["response"].content
'Exercises that can help with lower back pain include:\n\n- **Cat-Cow Stretch:** Start on your hands and knees, alternate between arching your back up (cat) and letting it sag down (cow). Do 10-15 repetitions.\n\n- **Bird Dog:** From hands and knees, extend opposite arm and leg while keeping your core engaged. Hold each side for about 5 seconds. Perform 10 repetitions per side.\n\n- **Partial Crunches:** Lie on your back with knees bent, cross arms over your chest, tighten your stomach muscles, and raise your shoulders off the floor. Hold briefly, then lower. Do 8-12 repetitions.\n\n- **Knee-to-Chest Stretch:** Lie on your back, pull one knee toward your chest while keeping the other foot flat. Hold for 15-30 seconds, then switch legs.\n\n- **Pelvic Tilts:** Lie on your back with knees bent, flatten your back against the floor by tightening your abs and tilting your pelvis slightly upward. Hold for 10 seconds and repeat 8-12 times.\n\nThese exercises are recommended to alleviate lower back discomfort and help prevent future episodes.'
multi_query_retrieval_chain.invoke({"question" : "How does sleep affect overall health?"})["response"].content
'Sleep plays a vital role in overall health by supporting physical, mental, and cognitive well-being. During sleep, the body repairs tissues, consolidates memories, and releases hormones that regulate growth and appetite. Adequate sleep—typically 7 to 9 hours per night—is crucial for maintaining a strong immune system, managing stress, and ensuring proper functioning of various bodily systems. Poor sleep or sleep disorders like insomnia can negatively impact health, leading to issues such as fatigue, weakened immunity, mental health problems, and increased risk of chronic conditions. Therefore, practicing good sleep hygiene and creating an optimal sleep environment are important strategies for promoting overall health.'
multi_query_retrieval_chain.invoke({"question" : "What are some natural remedies for stress and headaches?"})["response"].content
'Some natural remedies for stress and headaches include:\n\n- Deep breathing exercises (e.g., inhale for 4 counts, hold, exhale)\n- Progressive muscle relaxation (tensing and releasing muscle groups)\n- Grounding techniques (naming things you see, hear, feel, smell, taste)\n- Taking short walks, preferably in nature\n- Listening to calming music\n- Drinking water to stay hydrated\n- Applying cold or warm compresses to the head or neck\n- Resting in a dark, quiet room\n- Using essential oils such as peppermint or lavender\n- Maintaining a regular sleep schedule\n- Engaging in mindfulness and meditation practices\n- Managing stress through regular exercise, social support, and hobbies\n\nThese methods can help alleviate headaches and reduce stress naturally.'
❓ Question #2:
Explain how generating multiple reformulations of a user query can improve recall.

Answer:
Generating multiple reformulations of a user query improves recall because it increases the likelihood of matching relevant documents that use different wording, terminology, or phrasing than the original query.

In information retrieval, recall measures how many of the truly relevant documents in the corpus are successfully retrieved. If a user asks a single query in one specific phrasing, the retrieval system may only match documents that use similar vocabulary. However, many relevant documents may express the same concept differently.

For example, consider the user query: “How can I reduce high blood pressure naturally?”

Relevant documents might use alternative expressions such as: - “natural remedies for hypertension” - “lifestyle changes to lower blood pressure” - “non-pharmaceutical approaches to managing hypertension”

If the retrieval system searches only using the exact original wording, it may miss documents containing the term “hypertension” instead of “high blood pressure.” By generating multiple reformulations—such as replacing “high blood pressure” with “hypertension” or “reduce” with “manage” or “lower”—the system broadens the lexical and semantic search space.

This is especially important in lexical retrieval methods like BM25, where matching depends heavily on exact token overlap. Even in embedding-based retrieval, reformulations help because each reformulated query produces a slightly different embedding vector. Those variations can retrieve documents located in different regions of the embedding space, thereby covering more relevant material.

Conceptually, multi-query reformulation works like exploring a search space from multiple directions. Each reformulation acts as a different probe into the corpus. The union of results from all probes increases coverage of relevant documents, thereby improving recall.

However, this approach may also increase the number of irrelevant results, potentially lowering precision. That is why multi-query retrieval is often paired with reranking or contextual compression to filter and reorder the expanded candidate set.

In summary, generating multiple query reformulations improves recall by: - Reducing vocabulary mismatch - Capturing synonyms and alternative phrasing - Exploring different semantic interpretations - Expanding coverage across the document space

It systematically addresses one of the fundamental challenges in retrieval systems: the mismatch between how users express information needs and how documents express information content.

Task 8: Parent Document Retriever
A "small-to-big" strategy - the Parent Document Retriever works based on a simple strategy:

We split the full document into large "parent" chunks (e.g. 2000 characters).
Each parent chunk is further split into smaller "child" chunks (e.g. 400 characters).
The child chunks are stored in a VectorStore, while the parent chunks are stored in an in-memory docstore.
When we query our Retriever, we do a similarity search comparing our query vector to the child chunks.
Instead of returning the child chunks, we return their associated parent chunks.
The basic idea is:

Search for small, focused chunks (better semantic matching)
Return big chunks (richer surrounding context)
The intuition is that we're likely to find the most relevant information by limiting the amount of semantic information encoded in each embedding vector - but we're likely to miss relevant surrounding context if we only use that information.

Let's start by defining our parent and child splitters.

# from langchain.retrievers import ParentDocumentRetriever
# from langchain.storage import InMemoryStore
# from langchain_classic.retrievers import ParentDocumentRetriever

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from qdrant_client import QdrantClient, models

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
We'll need to set up a new QDrant vectorstore - and we'll use another useful pattern to do so!

NOTE: We are manually defining our embedding dimension, you'll need to change this if you're using a different embedding model.

from langchain_qdrant import QdrantVectorStore

client = QdrantClient(location=":memory:")

client.create_collection(
    collection_name="wellness_parent_child",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
)

parent_document_vectorstore = QdrantVectorStore(
    collection_name="wellness_parent_child", embedding=OpenAIEmbeddings(model="text-embedding-3-small"), client=client
)
Now we can create our InMemoryStore that will hold our "parent documents" - and build our retriever!

store = InMemoryStore()

parent_document_retriever = ParentDocumentRetriever(
    vectorstore=parent_document_vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
By default, this is empty as we haven't added any documents - let's add some now!

parent_document_retriever.add_documents(raw_docs, ids=None)
We'll create the same chain we did before - but substitute our new parent_document_retriever.

parent_document_retrieval_chain = (
    {"context": itemgetter("question") | parent_document_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
Let's give it a whirl!

parent_document_retrieval_chain.invoke({"question" : "What exercises can help with lower back pain?"})["response"].content
'Based on the information provided, exercises that can help with lower back pain include:\n\n- **Cat-Cow Stretch:** On hands and knees, alternate between arching your back (Cat) and letting it sag (Cow). Do 10-15 repetitions.\n- **Bird Dog:** On hands and knees, extend opposite arm and leg while engaging your core. Hold for 5 seconds, then switch sides. Do 10 repetitions per side.\n- **Partial Crunches:** Lie on your back with knees bent, cross arms over chest, tighten your stomach muscles, and lift your shoulders off the floor. Do 8-12 repetitions.\n- **Knee-to-Chest Stretch:** Lie on your back, pull one knee towards your chest, hold for 15-30 seconds, then switch legs.\n- **Pelvic Tilts:** Lie on your back with knees bent, tighten your abs, and tilt your pelvis up slightly to flatten your back against the floor. Hold for 10 seconds, repeat 8-12 times.\n\nThese gentle stretching and strengthening exercises can help alleviate lower back discomfort and may prevent future episodes.'
parent_document_retrieval_chain.invoke({"question" : "How does sleep affect overall health?"})["response"].content
'Sleep significantly affects overall health by supporting physical, mental, and cognitive functions. During sleep, the body repairs tissues, regenerates cells, and releases hormones that regulate growth and appetite. Adequate sleep—typically 7-9 hours per night—also aids in consolidating memories and maintaining mental well-being. Poor sleep or insufficient rest can lead to fatigue, decreased immune function, difficulty concentrating, and increased risk of health issues such as obesity, diabetes, and cardiovascular problems. Therefore, maintaining good sleep hygiene and ensuring restful sleep are essential for overall health.'
parent_document_retrieval_chain.invoke({"question" : "What are some natural remedies for stress and headaches?"})["response"].content
'Some natural remedies for stress and headaches include practicing deep breathing and progressive muscle relaxation, engaging in mindfulness and meditation, taking short walks in nature, listening to calming music, and doing light stretching or yoga. For headaches specifically, remedies such as staying well-hydrated, applying cold or warm compresses to the head or neck, resting in a dark, quiet room, massaging the temples and neck, and using essential oils like peppermint or lavender can be helpful.'
Overall, the performance seems largely the same. We can leverage a tool like Ragas to more effectively answer the question about the performance.

Task 9: Ensemble Retriever
In brief, an Ensemble Retriever simply takes 2, or more, retrievers and combines their retrieved documents based on a rank-fusion algorithm.

In this case - we're using the Reciprocal Rank Fusion algorithm.

Setting it up is as easy as providing a list of our desired retrievers - and the weights for each retriever.

#from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers import EnsembleRetriever


retriever_list = [bm25_retriever, naive_retriever, parent_document_retriever, compression_retriever, multi_query_retriever]
equal_weighting = [1/len(retriever_list)] * len(retriever_list)

ensemble_retriever = EnsembleRetriever(
    retrievers=retriever_list, weights=equal_weighting
)
We'll pack all of these retrievers together in an ensemble.

ensemble_retrieval_chain = (
    {"context": itemgetter("question") | ensemble_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
Let's look at our results!

ensemble_retrieval_chain.invoke({"question" : "What exercises can help with lower back pain?"})["response"].content
'Exercises that can help with lower back pain include:\n\n1. **Cat-Cow Stretch**: Start on your hands and knees, then alternate between arching your back up (cat) and letting it sag down (cow). Aim for 10-15 repetitions.\n\n2. **Bird Dog**: From hands and knees, extend opposite arm and leg while keeping your core engaged. Hold each extension for about 5 seconds, and do 10 repetitions per side.\n\n3. **Pelvic Tilts**: Lie on your back with knees bent, tighten your abs, and tilt your pelvis up slightly to flatten your back against the floor. Hold for 10 seconds, then repeat 8-12 times.\n\n4. **Partial Crunches**: Lie on your back with knees bent, cross arms over your chest, tighten your stomach muscles, and lift your shoulders off the floor briefly before lowering back down. Perform 8-12 repetitions.\n\n5. **Knee-to-Chest Stretch**: Lie on your back and pull one knee toward your chest while keeping the other foot flat on the floor. Hold for 15-30 seconds, then switch legs.\n\nThese exercises, performed gently and regularly, may help alleviate lower back discomfort and prevent future episodes. Remember to consult a healthcare professional before starting any new exercise routine, especially if you have existing health concerns.'
ensemble_retrieval_chain.invoke({"question" : "How does sleep affect overall health?"})["response"].content
'Sleep plays a vital role in overall health by supporting physical repair, mental well-being, and cognitive function. During sleep, the body repairs tissues, releases hormones that regulate growth and appetite, and consolidates memories. Adequate sleep (7-9 hours per night) helps strengthen the immune system, improve mood, and enhance learning and memory. Poor or insufficient sleep can lead to issues such as fatigue, stress, weakened immunity, and increased risk of chronic conditions. Maintaining good sleep hygiene and creating an optimal sleep environment are key strategies to promote high-quality sleep and overall health.'
ensemble_retrieval_chain.invoke({"question" : "What are some natural remedies for stress and headaches?"})["response"].content
'Some natural remedies for stress and headaches include:\n\n- For stress relief:\n  - Deep breathing exercises (inhale for 4 counts, hold for 4, exhale for 4)\n  - Progressive muscle relaxation (tensing and relaxing muscle groups)\n  - Grounding techniques (naming things you see, hear, feel, smell, and taste)\n  - Taking short walks, preferably in nature\n  - Listening to calming music\n\n- For headache relief:\n  - Drinking water and staying hydrated\n  - Applying cold or warm compresses to the head or neck\n  - Resting in a dark, quiet room\n  - Gently massaging temples and neck\n  - Using peppermint or lavender essential oils\n  - Consuming caffeine in small amounts (with caution)\n  - Maintaining a regular sleep schedule\n\nThese approaches can help manage stress and headaches naturally.'
Task 10: Semantic Chunking
While this is not a retrieval method - it is an effective way of increasing retrieval performance on corpora that have clean semantic breaks in them.

Essentially, Semantic Chunking is implemented by:

Embedding all sentences in the corpus.
Combining or splitting sequences of sentences based on their semantic similarity based on a number of possible thresholding methods:
percentile
standard_deviation
interquartile
gradient
Each sequence of related sentences is kept as a document!
Let's see how to implement this!

We'll use the percentile thresholding method for this example which will:

Calculate all distances between sentences, and then break apart sequences of setences that exceed a given percentile among all distances.

import sys
!{sys.executable} -m pip install -U langchain-experimental

from langchain_experimental.text_splitter import SemanticChunker

semantic_chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)
Now we can split our documents.

semantic_documents = semantic_chunker.split_documents(raw_docs)
Let's create a new vector store.

semantic_vectorstore = QdrantVectorStore.from_documents(
    semantic_documents,
    embeddings,
    location=":memory:",
    collection_name="wellness_guide_semantic_chunks"
)
We'll use naive retrieval for this example.

semantic_retriever = semantic_vectorstore.as_retriever(search_kwargs={"k" : 10})
Finally we can create our classic chain!

semantic_retrieval_chain = (
    {"context": itemgetter("question") | semantic_retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | chat_model, "context": itemgetter("context")}
)
And view the results!

semantic_retrieval_chain.invoke({"question" : "What exercises can help with lower back pain?"})["response"].content
'Exercises that can help with lower back pain include:\n\n- Cat-Cow Stretch: Start on hands and knees, alternate between arching your back up (cat) and letting it sag down (cow). Do 10-15 repetitions.\n- Partial Crunches: Lie on your back with knees bent, cross arms over chest, tighten stomach muscles and raise shoulders off the floor. Hold briefly, then lower. Do 8-12 repetitions.\n- Knee-to-Chest Stretch: Lie on your back, pull one knee toward your chest while keeping the other foot flat. Hold for 15-30 seconds, then switch legs.\n- Pelvic Tilts: Lie on your back with knees bent, flatten your back against the floor by tightening abs and tilting pelvis up slightly. Hold for 10 seconds, repeat 8-12 times.\n\nThese gentle stretching and strengthening exercises can help alleviate lower back discomfort and prevent future issues.'
semantic_retrieval_chain.invoke({"question" : "How does sleep affect overall health?"})["response"].content
'Sleep plays a vital role in overall health by supporting physical repair, mental well-being, and cognitive function. During sleep, the body repairs tissues, consolidates memories, and releases hormones that regulate growth and appetite. Adequate sleep—typically 7-9 hours per night for adults—helps maintain immune function, balance hormones, improve mood, and enhance learning and memory. Poor sleep quality or insufficient sleep can lead to increased fatigue, cognitive impairments, weakened immune defenses, and a higher risk for chronic conditions such as heart disease, diabetes, and mental health issues. Therefore, prioritizing good sleep hygiene and creating a conducive sleep environment are essential for overall health and wellness.'
semantic_retrieval_chain.invoke({"question" : "What are some natural remedies for stress and headaches?"})["response"].content
"Some natural remedies for stress and headaches include:\n\n- For stress:\n  - Deep breathing exercises (such as inhaling for 4 counts, holding for 4, exhaling for 4)\n  - Progressive muscle relaxation\n  - Grounding techniques (naming things you see, hear, feel, smell, and taste)\n  - Taking short walks in nature\n  - Listening to calming music\n  - Practicing mindfulness and meditation\n  - Engaging in hobbies and leisure activities\n  - Maintaining social connections and setting healthy boundaries\n\n- For headaches:\n  - Drinking plenty of water to stay hydrated\n  - Applying cold or warm compresses to the head or neck\n  - Resting in a dark, quiet room\n  - Gentle massage of temples and neck muscles\n  - Using essential oils like peppermint or lavender\n  - Maintaining a regular sleep schedule\n  - Managing triggers such as dehydration, stress, poor sleep, skipped meals, eye strain, or weather changes\n\nThese remedies focus on calming the mind and body naturally. However, if headaches or stress persist, it's best to consult with a healthcare professional."
❓ Question #3:
If sentences are short and highly repetitive (e.g., FAQs), how might semantic chunking behave, and how would you adjust the algorithm?

Answer:
If the sentences are short and highly repetitive (like FAQs), semantic chunking can behave “too conservatively” because adjacent sentence embeddings will look very similar. When everything is semantically near-identical, the similarity curve won’t show clear drops, so the algorithm may either (a) create very large chunks because it never detects a breakpoint, or (b) create unstable/random breakpoints driven by tiny embedding noise rather than real topic shifts. In practice, we can end up with chunks that are either bloated (hurting precision and increasing context noise) or fragmented in arbitrary places (hurting coherence).

To adjust, we want to add non-semantic constraints and stronger breakpoint signals. The simplest fix is to enforce a hard max chunk size (tokens/characters) and a minimum chunk size, so even if semantic similarity stays high, chunks don’t grow unbounded and don’t split too early. Next, we can tune the breakpoint sensitivity: we raise the breakpoint threshold so the chunker only splits on more meaningful similarity drops, or we switch from percentile-based thresholds to an absolute threshold calibrated on your FAQ corpus.

For FAQ-style text specifically, we can improve behavior by chunking around structure instead of pure semantics: treat each Q/A pair (or each heading + answer block) as an atomic unit, and then optionally merge multiple Q/A pairs until you hit a token budget. Another robust approach is to add lexical/format features as breakpoint triggers—e.g., split when a line matches Q: / A: patterns, question marks, numbering, or heading markers—because those signals are much more reliable than embeddings when content is repetitive. Finally, if repetition is extreme, we can consider deduplication or near-duplicate clustering before chunking, so the chunker isn’t forced to “find” boundaries that don’t exist semantically.

🤝 Breakout Room Part #2
🏗️ Activity #1:
Your task is to evaluate the various Retriever methods against each other.

You are expected to:

Create a "golden dataset"
Use Synthetic Data Generation (powered by Ragas, or otherwise) to create this dataset
Evaluate each retriever with retriever specific Ragas metrics
Semantic Chunking is not considered a retriever method and will not be required for marks, but you may find it useful to do a "semantic chunking on" vs. "semantic chunking off" comparison between them
Compile these in a list and write a small paragraph about which is best for this particular data and why.
Your analysis should factor in:

Cost
Latency
Performance
NOTE: This is NOT required to be completed in class. Please spend time in your breakout rooms creating a plan before moving on to writing code.

HINTS:
LangSmith provides detailed information about latency and cost.
Step 0 — Install deps

import sys
!{sys.executable} -m pip install -U ragas datasets pandas numpy

!{sys.executable} -m pip install -U langsmith
Step 1 — Create a golden dataset using Ragas synthetic generation

This generates (question, reference_answer, reference_contexts) based on your docs.

import json, random
import pandas as pd

# Make runs repeatable
random.seed(42)

PROMPT = """You are generating a golden evaluation dataset for a RAG system.
Given the context, create ONE evaluation item.

Rules:
- The question MUST be answerable ONLY from the context.
- The answer MUST be short, factual, and directly supported by the context.
- Avoid vague questions. Prefer questions that require specific details.
Return STRICT JSON with keys: question, ground_truth.

CONTEXT:
{context}
"""

def llm_make_gold_dataset(docs, n=30, max_chars=2500):
    # sample across chunks to cover the corpus
    if len(docs) <= n:
        picked = docs
    else:
        idxs = sorted(random.sample(range(len(docs)), n))
        picked = [docs[i] for i in idxs]

    rows = []
    for d in picked:
        context = d.page_content[:max_chars]
        resp = chat_model.invoke(PROMPT.format(context=context)).content.strip()

        # robust JSON extraction
        resp_clean = resp.strip().strip("```").replace("json", "").strip()
        try:
            data = json.loads(resp_clean)
            rows.append({
                "question": data["question"].strip(),
                "ground_truth": data["ground_truth"].strip(),
            })
        except Exception:
            # If parsing fails, skip this row
            continue

    return pd.DataFrame(rows)

gold_df = llm_make_gold_dataset(wellness_docs, n=30)
print("gold rows:", len(gold_df))
gold_df.head(3)
gold rows: 27
question	ground_truth
0	What are two benefits of regular physical acti...	Improve brain health and manage weight
1	According to the context, how many days per we...	2 or more days per week
2	What percentage of adults experience lower bac...	Approximately 80%
Step 2 — Retrieve contexts per retriever + measure latency

import time
from datasets import Dataset

def retrieve_docs(retriever, query: str):
    # Runnable-style (newer)
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    # Classic LC retriever
    if hasattr(retriever, "get_relevant_documents"):
        return retriever.get_relevant_documents(query)
    raise TypeError(f"Retriever unsupported: {type(retriever)}")

def run_retriever(retriever, gold_df: pd.DataFrame):
    rows = []
    latencies = []

    for q, gt in gold_df[["question", "ground_truth"]].itertuples(index=False):
        t0 = time.perf_counter()
        docs = retrieve_docs(retriever, q)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # ms
        contexts = [d.page_content for d in docs] if docs else []

        rows.append({
            "question": q,
            "ground_truth": gt,
            "contexts": contexts
        })

    out_df = pd.DataFrame(rows)

    latency = {
        "p50_ms": float(pd.Series(latencies).quantile(0.50)),
        "p95_ms": float(pd.Series(latencies).quantile(0.95)),
        "avg_ms": float(pd.Series(latencies).mean()),
    }
    return out_df, latency
Step 3 — Ragas evaluation (context precision + context recall)

from ragas import evaluate

try:
    from ragas.metrics import context_precision, context_recall
except Exception:
    raise ImportError("Your ragas install does not expose context_precision/context_recall. Run: !pip show ragas")

def ragas_retriever_eval(df_with_contexts: pd.DataFrame):
    ds = Dataset.from_pandas(df_with_contexts[["question", "ground_truth", "contexts"]])
    results = evaluate(ds, metrics=[context_precision, context_recall])
    return dict(results)
/var/folders/w5/khbwggzj1m54r941_dyt49vw0000gn/T/ipykernel_10059/3843123485.py:4: DeprecationWarning: Importing context_precision from 'ragas.metrics' is deprecated and will be removed in v1.0. Please use 'ragas.metrics.collections' instead. Example: from ragas.metrics.collections import context_precision
  from ragas.metrics import context_precision, context_recall
/var/folders/w5/khbwggzj1m54r941_dyt49vw0000gn/T/ipykernel_10059/3843123485.py:4: DeprecationWarning: Importing context_recall from 'ragas.metrics' is deprecated and will be removed in v1.0. Please use 'ragas.metrics.collections' instead. Example: from ragas.metrics.collections import context_recall
  from ragas.metrics import context_precision, context_recall
Step 4 — Run eval across all retrievers + build a results table

import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from datasets import Dataset

def ragas_retriever_eval(df_with_contexts: pd.DataFrame):
    ds = Dataset.from_pandas(df_with_contexts[["question", "ground_truth", "contexts"]])

    result = evaluate(
        ds,
        metrics=[context_precision, context_recall],
        llm=chat_model,
        embeddings=embeddings
    )

    # version-safe extraction
    if hasattr(result, "to_pandas"):
        # best: contains per-sample + summary
        return result.to_pandas()
    if hasattr(result, "scores"):
        return result.scores
    if hasattr(result, "dict"):
        return result.dict()
    # last resort: print object
    return result

mini_gold = gold_df.head(3)

out_df, latency = run_retriever(naive_retriever, mini_gold)
print("retrieved rows:", len(out_df), "avg latency(ms):", latency["avg_ms"])

scores = ragas_retriever_eval(out_df)
print(scores)
/var/folders/w5/khbwggzj1m54r941_dyt49vw0000gn/T/ipykernel_10059/2964912783.py:2: DeprecationWarning: Importing context_precision from 'ragas.metrics' is deprecated and will be removed in v1.0. Please use 'ragas.metrics.collections' instead. Example: from ragas.metrics.collections import context_precision
  from ragas.metrics import context_precision, context_recall
/var/folders/w5/khbwggzj1m54r941_dyt49vw0000gn/T/ipykernel_10059/2964912783.py:2: DeprecationWarning: Importing context_recall from 'ragas.metrics' is deprecated and will be removed in v1.0. Please use 'ragas.metrics.collections' instead. Example: from ragas.metrics.collections import context_recall
  from ragas.metrics import context_precision, context_recall
retrieved rows: 3 avg latency(ms): 251.0543609581267
Evaluating: 100%|██████████| 6/6 [00:14<00:00,  2.46s/it]
                                          user_input  \
0  What are two benefits of regular physical acti...   
1  According to the context, how many days per we...   
2  What percentage of adults experience lower bac...   

                                  retrieved_contexts  \
0  [The Personal Wellness Guide\nA Comprehensive ...   
1  [The four main types of exercise are aerobic (...   
2  [Chapter 2: Exercises for Common Problems\n\nL...   

                                reference  context_precision  context_recall  
0  Improve brain health and manage weight              0.625             1.0  
1                 2 or more days per week              1.000             1.0  
2                       Approximately 80%              1.000             1.0  
import time
import pandas as pd

# --- Step 4: Evaluate retrievers (rate-limit safe) ---

retrievers = {
    "naive_vector": naive_retriever,
    "bm25": bm25_retriever,
    "parent_doc": parent_document_retriever,
    # Cohere rerank + ensemble will hit Cohere trial rate limits (10 calls/min)
    "rerank_contextual_compression": compression_retriever,
    "multi_query": multi_query_retriever,
    "ensemble_rrf": ensemble_retriever,
}

# Rate-limit control:
# - Cohere trial: max ~10 calls/min => ~1 call every 6+ seconds.
COHERE_DELAY_SECONDS = 7
DEFAULT_DELAY_SECONDS = 0

def extract_summary(scores_obj):
    # Your ragas_retriever_eval returns a DataFrame in your setup
    if isinstance(scores_obj, pd.DataFrame):
        needed = ["context_precision", "context_recall"]
        missing = [c for c in needed if c not in scores_obj.columns]
        if missing:
            raise ValueError(f"Missing expected ragas metric columns: {missing}. Got: {list(scores_obj.columns)}")
        return {c: float(scores_obj[c].mean()) for c in needed}
    if isinstance(scores_obj, dict):
        out = {}
        for k in ["context_precision", "context_recall"]:
            if k in scores_obj:
                out[k] = float(scores_obj[k])
        return out
    raise TypeError(f"Unexpected scores object type: {type(scores_obj)}")

def run_retriever_with_sleep(retriever, gold_df, delay_seconds=0):
    rows = []
    latencies = []

    for q, gt in gold_df[["question", "ground_truth"]].itertuples(index=False):
        t0 = time.perf_counter()
        docs = retrieve_docs(retriever, q)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)
        contexts = [d.page_content for d in docs] if docs else []

        rows.append({"question": q, "ground_truth": gt, "contexts": contexts})

        if delay_seconds and delay_seconds > 0:
            time.sleep(delay_seconds)

    out_df = pd.DataFrame(rows)
    latency = {
        "p50_ms": float(pd.Series(latencies).quantile(0.50)),
        "p95_ms": float(pd.Series(latencies).quantile(0.95)),
        "avg_ms": float(pd.Series(latencies).mean()),
    }
    return out_df, latency

all_results = []
skipped = []

for name, r in retrievers.items():
    try:
        # Apply sleep only for Cohere rerank to respect trial limit
        delay = COHERE_DELAY_SECONDS if name == "rerank_contextual_compression" else DEFAULT_DELAY_SECONDS

        # Ensemble includes rerank inside it (and will still 429 with trial key), so skip it
        if name == "ensemble_rrf":
            skipped.append({"retriever": name, "reason": "Skipped to avoid Cohere trial rate limit (ensemble triggers rerank calls)."})
            continue

        out_df, latency = run_retriever_with_sleep(r, gold_df, delay_seconds=delay)

        # Ragas evaluation (may also call LLM; keep it as-is)
        scores_obj = ragas_retriever_eval(out_df)
        scores = extract_summary(scores_obj)

        all_results.append({"retriever": name, **scores, **latency})

    except Exception as e:
        skipped.append({"retriever": name, "reason": str(e)})

results_df = pd.DataFrame(all_results).sort_values(
    by=["context_recall", "context_precision"],
    ascending=False
)

print("=== RESULTS ===")
display(results_df)

print("\n=== SKIPPED / FAILED ===")
display(pd.DataFrame(skipped))
Evaluating: 100%|██████████| 54/54 [02:07<00:00,  2.37s/it]
Evaluating: 100%|██████████| 54/54 [00:58<00:00,  1.08s/it]
Evaluating: 100%|██████████| 54/54 [00:39<00:00,  1.35it/s]
Evaluating:  48%|████▊     | 26/54 [00:28<00:41,  1.49s/it]Prompt fix_output_format failed to parse output: The output parser failed to parse the output including retries.
Prompt fix_output_format failed to parse output: The output parser failed to parse the output including retries.
Prompt fix_output_format failed to parse output: The output parser failed to parse the output including retries.
Prompt context_recall_classification_prompt failed to parse output: The output parser failed to parse the output including retries.
Exception raised in Job[3]: RagasOutputParserException(The output parser failed to parse the output including retries.)
Evaluating: 100%|██████████| 54/54 [00:47<00:00,  1.15it/s]
Evaluating: 100%|██████████| 54/54 [02:48<00:00,  3.12s/it]
=== RESULTS ===
retriever	context_precision	context_recall	p50_ms	p95_ms	avg_ms
4	multi_query	0.837009	0.973937	1369.499667	1781.116708	1425.830927
2	parent_doc	0.993827	0.950414	183.832542	382.268866	203.387877
3	rerank_contextual_compression	0.972222	0.939103	432.125167	874.733004	495.161960
0	naive_vector	0.836464	0.930748	215.363208	429.665588	242.303363
1	bm25	0.601852	0.875162	0.214000	0.370771	0.258108
=== SKIPPED / FAILED ===
retriever	reason
0	ensemble_rrf	Skipped to avoid Cohere trial rate limit (ense...
Step 5 — Add cost estimate

# Step 5 — Add cost estimate
cost_rank = {
    "bm25": "Low (no LLM, no rerank)",
    "naive_vector": "Low (1 vector search)",
    "parent_doc": "Low–Medium (vector search + parent fetch)",
    "ensemble_rrf": "Medium–High (multiple retrievals per query)",
    "multi_query": "High (LLM generates multiple queries + multiple retrievals)",
    "rerank_contextual_compression": "High (Cohere rerank API call per query)"
}

results_df["cost_estimate"] = results_df["retriever"].map(cost_rank)
results_df
retriever	context_precision	context_recall	p50_ms	p95_ms	avg_ms	cost_estimate
4	multi_query	0.837009	0.973937	1369.499667	1781.116708	1425.830927	High (LLM generates multiple queries + multipl...
2	parent_doc	0.993827	0.950414	183.832542	382.268866	203.387877	Low–Medium (vector search + parent fetch)
3	rerank_contextual_compression	0.972222	0.939103	432.125167	874.733004	495.161960	High (Cohere rerank API call per query)
0	naive_vector	0.836464	0.930748	215.363208	429.665588	242.303363	Low (1 vector search)
1	bm25	0.601852	0.875162	0.214000	0.370771	0.258108	Low (no LLM, no rerank)
Step 6 — Final conclusion paragraph

# Step 6 — Final conclusion paragraph
best = results_df.iloc[0]
best_name = best["retriever"]

print(f"""
On the HealthWellnessGuide dataset, {best_name} performed best overall by achieving the strongest context recall while maintaining good context precision, indicating it retrieves the needed supporting passages with relatively low noise. 
From a latency standpoint, the fastest approaches were typically BM25 and naive vector retrieval because they require a single retrieval step, while multi-query and reranking methods were slower due to extra LLM/reranker calls. 
From a cost perspective, BM25 and naive retrieval are lowest-cost, while multi-query and contextual compression are highest-cost because they add model calls per query. 
Overall, {best_name} is the best fit for this corpus because it balances coverage of relevant content with reasonable noise, while the more expensive methods only provide incremental gains relative to their additional latency and cost.
""")
On the HealthWellnessGuide dataset, multi_query performed best overall by achieving the strongest context recall while maintaining good context precision, indicating it retrieves the needed supporting passages with relatively low noise. 
From a latency standpoint, the fastest approaches were typically BM25 and naive vector retrieval because they require a single retrieval step, while multi-query and reranking methods were slower due to extra LLM/reranker calls. 
From a cost perspective, BM25 and naive retrieval are lowest-cost, while multi-query and contextual compression are highest-cost because they add model calls per query. 
Overall, multi_query is the best fit for this corpus because it balances coverage of relevant content with reasonable noise, while the more expensive methods only provide incremental gains relative to their additional latency and cost.

import ragas, pkgutil
print(ragas.__version__)
[m.name for m in pkgutil.iter_modules(ragas.__path__, ragas.__name__ + ".") if "test" in m.name]
fastest = results_df.sort_values("avg_ms").iloc[0]["retriever"]
print("fastest:", fastest)
0.4.3
fastest: bm25
Advanced Build: Reproduce the RAGAS Synthetic Data Generation Steps - but utilize a LangGraph Agent Graph, instead of the Knowledge Graph approach.

This generation should leverage the Evol Instruct method to generate synthetic data.

Your final state (output) should contain (at least, not limited to):

List(dict): Evolved Questions, their IDs, and their Evolution Type. List(dict): Question IDs, and Answer to the referenced Evolved Question. List(dict): Question IDs, and the relevant Context(s) to the Evolved Question. The Graph should handle:

Simple Evolution. Multi-Context Evolution. Reasoning Evolution. It should take, as input, a list of LangChain Documents.

Step A — Dependencies (keep it minimal)

import sys
!{sys.executable} -m pip install -U langgraph langchain langchain-openai pydantic
Step B — Imports + models

import os, uuid, json, random
from typing import List, Dict, Literal, TypedDict, Optional

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langgraph.graph import StateGraph, END

random.seed(42)

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
Step C — Define the output schemas (this kills debugging)

#1) Evolved question schema
EvolutionType = Literal["simple", "multi_context", "reasoning"]

class EvolvedQuestion(BaseModel):
    question_id: str = Field(..., description="Unique ID for this evolved question")
    evolution_type: EvolutionType
    question: str

#2) Answer schema
class AnswerItem(BaseModel):
    question_id: str
    answer: str

# 3) Context schema
class ContextItem(BaseModel):
    question_id: str
    contexts: List[str]
Step D — Prompts (Evol-Instruct style, but controlled)

We’ll generate questions from context(s), then answer them using only those context(s).

# Prompt 1: Simple evolution
simple_prompt = ChatPromptTemplate.from_template("""
You are generating an evaluation question for a RAG system.

Given the context, write ONE specific question that is directly answerable from this context alone.
Avoid vague questions.

Return JSON with keys: question

CONTEXT:
{context}
""")

# Prompt 2: Multi-context evolution
multi_context_prompt = ChatPromptTemplate.from_template("""
You are generating an evaluation question that requires combining information from multiple contexts.

Write ONE question that requires BOTH contexts to answer correctly.
Return JSON with keys: question

CONTEXT A:
{context_a}

CONTEXT B:
{context_b}
""")

# Prompt 3: Reasoning evolution
reasoning_prompt = ChatPromptTemplate.from_template("""
You are generating an evaluation question that requires reasoning based on the context.

Write ONE question where the answer is not a direct quote, but can be logically inferred from the context.
Return JSON with keys: question

CONTEXT:
{context}
""")

# Prompt 4: Answering (ground truth)

answer_prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided contexts.
If the contexts do not contain enough information, respond exactly with: "I don't know."

Return JSON with keys: answer

QUESTION:
{question}

CONTEXTS:
{contexts}
""")

q_parser = JsonOutputParser()
a_parser = JsonOutputParser()
Step E — LangGraph State + Nodes

class GenState(TypedDict):
    docs: List[Document]
    n_simple: int
    n_multi: int
    n_reasoning: int

    evolved_questions: List[Dict]
    answers: List[Dict]
    contexts: List[Dict]


# Utility: sample docs

def pick_doc(docs: List[Document]) -> Document:
    return random.choice(docs)

def pick_two_docs(docs: List[Document]) -> List[Document]:
    a = random.choice(docs)
    b = random.choice(docs)
    while b == a and len(docs) > 1:
        b = random.choice(docs)
    return [a, b]

# Node 1: Generate evolved questions
def generate_questions(state: GenState) -> GenState:
    docs = state["docs"]

    evolved = []

    # simple
    for _ in range(state["n_simple"]):
        d = pick_doc(docs)
        resp = llm.invoke(simple_prompt.format_messages(context=d.page_content))
        q = q_parser.parse(resp.content)["question"].strip()
        evolved.append(EvolvedQuestion(
            question_id=str(uuid.uuid4()),
            evolution_type="simple",
            question=q
        ).model_dump())

    # reasoning
    for _ in range(state["n_reasoning"]):
        d = pick_doc(docs)
        resp = llm.invoke(reasoning_prompt.format_messages(context=d.page_content))
        q = q_parser.parse(resp.content)["question"].strip()
        evolved.append(EvolvedQuestion(
            question_id=str(uuid.uuid4()),
            evolution_type="reasoning",
            question=q
        ).model_dump())

    # multi-context
    for _ in range(state["n_multi"]):
        a, b = pick_two_docs(docs)
        resp = llm.invoke(multi_context_prompt.format_messages(
            context_a=a.page_content,
            context_b=b.page_content
        ))
        q = q_parser.parse(resp.content)["question"].strip()
        evolved.append(EvolvedQuestion(
            question_id=str(uuid.uuid4()),
            evolution_type="multi_context",
            question=q
        ).model_dump())

    state["evolved_questions"] = evolved
    return state

# Node 2: Attach reference contexts to each question

def attach_contexts(state: GenState) -> GenState:
    docs = state["docs"]
    ctx_items = []

    for item in state["evolved_questions"]:
        qid = item["question_id"]
        et = item["evolution_type"]

        if et in ["simple", "reasoning"]:
            d = pick_doc(docs)
            ctx_items.append(ContextItem(
                question_id=qid,
                contexts=[d.page_content]
            ).model_dump())

        else:  # multi_context
            a, b = pick_two_docs(docs)
            ctx_items.append(ContextItem(
                question_id=qid,
                contexts=[a.page_content, b.page_content]
            ).model_dump())

    state["contexts"] = ctx_items
    return state

# Node 3: Generate ground-truth answers using only contexts

def generate_answers(state: GenState) -> GenState:
    ctx_by_id = {c["question_id"]: c["contexts"] for c in state["contexts"]}
    ans_items = []

    for q in state["evolved_questions"]:
        qid = q["question_id"]
        question = q["question"]
        contexts = ctx_by_id[qid]

        resp = llm.invoke(answer_prompt.format_messages(
            question=question,
            contexts="\n\n---\n\n".join(contexts)
        ))
        answer = a_parser.parse(resp.content)["answer"].strip()

        ans_items.append(AnswerItem(
            question_id=qid,
            answer=answer
        ).model_dump())

    state["answers"] = ans_items
    return state
Step F — Build + run the graph

graph = StateGraph(GenState)

graph.add_node("generate_questions", generate_questions)
graph.add_node("attach_contexts", attach_contexts)
graph.add_node("generate_answers", generate_answers)

graph.set_entry_point("generate_questions")
graph.add_edge("generate_questions", "attach_contexts")
graph.add_edge("attach_contexts", "generate_answers")
graph.add_edge("generate_answers", END)

app = graph.compile()

# Run it

inputs = {
    "docs": wellness_docs,     # or raw_docs if you want
    "n_simple": 10,
    "n_multi": 10,
    "n_reasoning": 10,
    "evolved_questions": [],
    "answers": [],
    "contexts": [],
}

final_state = app.invoke(inputs)

evolved_questions = final_state["evolved_questions"]
answers = final_state["answers"]
contexts = final_state["contexts"]

print(len(evolved_questions), len(answers), len(contexts))
evolved_questions[0], answers[0], contexts[0]
30 30 30
({'question_id': 'baaf1212-5761-4b8d-ba5c-f5377a54f988',
  'evolution_type': 'simple',
  'question': 'According to the sample day of balanced eating, what beverage is consumed in the evening?'},
 {'question_id': 'baaf1212-5761-4b8d-ba5c-f5377a54f988',
  'answer': "I don't know."},
 {'question_id': 'baaf1212-5761-4b8d-ba5c-f5377a54f988',
  'contexts': ['Strategies for better balance:\n- Set clear boundaries between work and personal time\n- Learn to say no to non-essential commitments\n- Schedule personal time like you would meetings\n- Take regular breaks throughout the day\n- Use vacation time\n- Delegate when possible\n- Disconnect from work emails/calls after hours\n\nChapter 20: Social Connections and Health\n\nStrong social connections are linked to better mental and physical health, increased longevity, and greater happiness.']})
Step G — The deliverable objects (exact formats required) - evolved_questions → List[dict] with ids + evolution type - answers → List[dict] with ids + answers - contexts → List[dict] with ids + contexts

import json

with open("evolved_questions.json", "w") as f:
    json.dump(evolved_questions, f, indent=2)

with open("answers.json", "w") as f:
    json.dump(answers, f, indent=2)

with open("contexts.json", "w") as f:
    json.dump(contexts, f, indent=2)