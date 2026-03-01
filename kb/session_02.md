
Your First RAG Application: Personal Wellness Assistant
In this notebook, we'll walk you through each of the components that are involved in a simple RAG application by building a Personal Wellness Assistant.

Imagine having an AI assistant that can answer your health and wellness questions based on a curated knowledge base - that's exactly what we'll build here. We won't be leveraging any fancy tools, just the OpenAI Python SDK, Numpy, and some classic Python.

NOTE: This was done with Python 3.12.3.

NOTE: There might be compatibility issues if you're on NVIDIA driver >552.44 As an interim solution - you can rollback your drivers to the 552.44.

Table of Contents:
Task 1: Imports and Utilities
Task 2: Documents (Loading our Wellness Knowledge Base)
Task 3: Embeddings and Vectors
Task 4: Prompts
Task 5: Retrieval Augmented Generation (Building the Wellness Assistant)
🚧 Activity #1: Enhance Your Wellness Assistant
Let's look at a rather complicated looking visual representation of a basic RAG application.



Task 1: Imports and Utilities
We're just doing some imports and enabling async to work within the Jupyter environment here, nothing too crazy!

from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import asyncio
import nest_asyncio
nest_asyncio.apply()
Task 2: Documents
We'll be concerning ourselves with this part of the flow in the following section:



Loading Source Documents
So, first things first, we need some documents to work with.

While we could work directly with the .txt files (or whatever file-types you wanted to extend this to) we can instead do some batch processing of those documents at the beginning in order to store them in a more machine compatible format.

In this case, we're going to parse our text file into a single document in memory.

Let's look at the relevant bits of the TextFileLoader class:

def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())
We're simply loading the document using the built in open method, and storing that output in our self.documents list.

NOTE: We're using a comprehensive Health & Wellness Guide as our sample data. This content covers exercise, nutrition, sleep, stress management, and healthy habits - perfect for building a personal wellness assistant!

text_loader = TextFileLoader("data/HealthWellnessGuide.txt")
documents = text_loader.load_documents()
len(documents)
1
print(documents[0][:100])
The Personal Wellness Guide
A Comprehensive Resource for Health and Well-being

PART 1: EXERCISE AND
Splitting Text Into Chunks
As we can see, there is one massive document.

We'll want to chunk the document into smaller parts so it's easier to pass the most relevant snippets to the LLM.

There is no fixed way to split/chunk documents - and you'll need to rely on some intuition as well as knowing your data very well in order to build the most robust system.

For this toy example, we'll just split blindly on length.

There's an opportunity to clear up some terminology here, for this course we will stick to the following:

"source documents" : The .txt, .pdf, .html, ..., files that make up the files and information we start with in its raw format
"document(s)" : single (or more) text object(s)
"corpus" : the combination of all of our documents
As you can imagine (though it's not specifically true in this toy example) the idea of splitting documents is to break them into manageable sized chunks that retain the most relevant local context.

text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)
len(split_documents)
21
Let's take a look at some of the documents we've managed to split.

split_documents[0:1]
['The Personal Wellness Guide\nA Comprehensive Resource for Health and Well-being\n\nPART 1: EXERCISE AND MOVEMENT\n\nChapter 1: Understanding Exercise Basics\n\nExercise is one of the most important things you can do for your health. Regular physical activity can improve your brain health, help manage weight, reduce the risk of disease, strengthen bones and muscles, and improve your ability to do everyday activities.\n\nThe four main types of exercise are aerobic (cardio), strength training, flexibility, and balance exercises. A well-rounded fitness routine includes all four types. Adults should aim for at least 150 minutes of moderate-intensity aerobic activity per week, along with muscle-strengthening activities on 2 or more days per week.\n\nChapter 2: Exercises for Common Problems\n\nLower Back Pain Relief\nLower back pain affects approximately 80% of adults at some point in their lives. Gentle stretching and strengthening exercises can help alleviate discomfort and prevent future episodes.\n\nReco']
Task 3: Embeddings and Vectors
Next, we have to convert our corpus into a "machine readable" format as we explored in the Embedding Primer notebook.

Today, we're going to talk about the actual process of creating, and then storing, these embeddings, and how we can leverage that to intelligently add context to our queries.

OpenAI API Key
In order to access OpenAI's APIs, we'll need to provide our OpenAI API Key!

You can work through the folder "OpenAI API Key Setup" for more information on this process if you don't already have an API Key!

import os
import openai
from getpass import getpass

openai.api_key = getpass("OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = openai.api_key
Vector Database
Let's set up our vector database to hold all our documents and their embeddings!

While this is all baked into 1 call - we can look at some of the code that powers this process to get a better understanding:

Let's look at our VectorDatabase().__init__():

def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()
As you can see - our vectors are merely stored as a dictionary of np.array objects.

Secondly, our VectorDatabase() has a default EmbeddingModel() which is a wrapper for OpenAI's text-embedding-3-small model.

Quick Info About text-embedding-3-small:

It has a context window of 8191 tokens
It returns vectors with dimension 1536
❓Question #1:
The default embedding dimension of text-embedding-3-small is 1536, as noted above.

Is there any way to modify this dimension?
What technique does OpenAI use to achieve this?
NOTE: Check out this API documentation for the answer to question #1.1, and this paper for an answer to question #1.2!

✅ Answer:
Yes. The OpenAI Embeddings API allows you to explicitly request a smaller embedding dimension using the optional dimensions parameter when creating embeddings.
From the API reference document : The dimensions field lets you control how many dimensions the output embedding should have. This is supported for text-embedding-3 models and later. The default for text-embedding-3-small is 1536, but you can request fewer (e.g. 256, 512, 1024). This is documented directly in the Embeddings API specification.

OpenAI uses a technique called Matryoshka Representation Learning (MRL). How this is done? The embedding is trained so that earlier dimensions carry the most semantic information. Later dimensions add progressively finer detail. You can truncate the vector (use fewer dimensions) and still preserve meaning.
This allows smaller vectors with reasonable semantic quality, faster similarity search and lower storage and memory usage

The technique is described in the Matryoshka Representation Learning paper referenced in the notebook.

We can call the async_get_embeddings method of our EmbeddingModel() on a list of str and receive a list of float back!

async def async_get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        return await aget_embeddings(
            list_of_text=list_of_text, engine=self.embeddings_model_name
        )
We cast those to np.array when we build our VectorDatabase():

async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self
And that's all we need to do!

vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))
❓Question #2:
What are the benefits of using an async approach to collecting our embeddings?

NOTE: Determining the core difference between async and sync will be useful! If you get stuck - ask ChatGPT!

✅ Answer: Using an async approach allows multiple embedding API requests to be handled concurrently instead of sequentially, reducing total processing time, improving efficiency for I/O-bound operations, and making the system scalable for large document sets.
So, to review what we've done so far in natural language:

We load source documents
We split those source documents into smaller chunks (documents)
We send each of those documents to the text-embedding-3-small OpenAI API endpoint
We store each of the text representations with the vector representations as keys/values in a dictionary
Semantic Similarity
The next step is to be able to query our VectorDatabase() with a str and have it return to us vectors and text that is most relevant from our corpus.

We're going to use the following process to achieve this in our toy example:

We need to embed our query with the same EmbeddingModel() as we used to construct our VectorDatabase()
We loop through every vector in our VectorDatabase() and use a distance measure to compare how related they are
We return a list of the top k closest vectors, with their text representations
There's some very heavy optimization that can be done at each of these steps - but let's just focus on the basic pattern in this notebook.

We are using cosine similarity as a distance metric in this example - but there are many many distance metrics you could use - like these

We are using a rather inefficient way of calculating relative distance between the query vector and all other vectors - there are more advanced approaches that are much more efficient, like ANN

vector_db.search_by_text("What exercises help with lower back pain?", k=3)
[(' Relief\nLower back pain affects approximately 80% of adults at some point in their lives. Gentle stretching and strengthening exercises can help alleviate discomfort and prevent future episodes.\n\nRecommended exercises for lower back pain include:\n- Cat-Cow Stretch: Start on hands and knees, alternate between arching your back up (cat) and letting it sag down (cow). Do 10-15 repetitions.\n- Bird Dog: From hands and knees, extend opposite arm and leg while keeping your core engaged. Hold for 5 seconds, then switch sides. Do 10 repetitions per side.\n- Partial Crunches: Lie on your back with knees bent, cross arms over chest, tighten stomach muscles and raise shoulders off floor. Hold briefly, then lower. Do 8-12 repetitions.\n- Knee-to-Chest Stretch: Lie on your back, pull one knee toward your chest while keeping the other foot flat. Hold for 15-30 seconds, then switch legs.\n- Pelvic Tilts: Lie on your back with knees bent, flatten your back against the floor by tightening abs and tilting p',
  np.float64(0.7478003643039624)),
 ('The Personal Wellness Guide\nA Comprehensive Resource for Health and Well-being\n\nPART 1: EXERCISE AND MOVEMENT\n\nChapter 1: Understanding Exercise Basics\n\nExercise is one of the most important things you can do for your health. Regular physical activity can improve your brain health, help manage weight, reduce the risk of disease, strengthen bones and muscles, and improve your ability to do everyday activities.\n\nThe four main types of exercise are aerobic (cardio), strength training, flexibility, and balance exercises. A well-rounded fitness routine includes all four types. Adults should aim for at least 150 minutes of moderate-intensity aerobic activity per week, along with muscle-strengthening activities on 2 or more days per week.\n\nChapter 2: Exercises for Common Problems\n\nLower Back Pain Relief\nLower back pain affects approximately 80% of adults at some point in their lives. Gentle stretching and strengthening exercises can help alleviate discomfort and prevent future episodes.\n\nReco',
  np.float64(0.4567084158663246)),
 ('chest while keeping the other foot flat. Hold for 15-30 seconds, then switch legs.\n- Pelvic Tilts: Lie on your back with knees bent, flatten your back against the floor by tightening abs and tilting pelvis up slightly. Hold for 10 seconds, repeat 8-12 times.\n\nNeck and Shoulder Tension\nDesk work and poor posture often lead to neck and shoulder tension. These exercises can provide relief:\n- Neck Rolls: Slowly roll your head in a circle, 5 times in each direction.\n- Shoulder Shrugs: Raise shoulders toward ears, hold for 5 seconds, then release. Repeat 10 times.\n- Chest Opener: Clasp hands behind back, squeeze shoulder blades together, and lift arms slightly. Hold for 15-30 seconds.\n- Chin Tucks: While sitting or standing tall, pull your chin back to create a "double chin." Hold for 5 seconds, repeat 10 times.\n\nChapter 3: Building a Workout Routine\n\nStarting a new exercise routine can feel overwhelming. The key is to start slowly and gradually increase intensity and duration over time.\n\nBe',
  np.float64(0.43536130238098786))]
Task 4: Prompts
In the following section, we'll be looking at the role of prompts - and how they help us to guide our application in the right direction.

In this notebook, we're going to rely on the idea of "zero-shot in-context learning".

This is a lot of words to say: "We will ask it to perform our desired task in the prompt, and provide no examples."

XYZRolePrompt
Before we do that, let's stop and think a bit about how OpenAI's chat models work.

We know they have roles - as is indicated in the following API documentation

There are three roles, and they function as follows (taken directly from OpenAI):

{"role" : "system"} : The system message helps set the behavior of the assistant. For example, you can modify the personality of the assistant or provide specific instructions about how it should behave throughout the conversation. However note that the system message is optional and the model’s behavior without a system message is likely to be similar to using a generic message such as "You are a helpful assistant."
{"role" : "user"} : The user messages provide requests or comments for the assistant to respond to.
{"role" : "assistant"} : Assistant messages store previous assistant responses, but can also be written by you to give examples of desired behavior.
The main idea is this:

You start with a system message that outlines how the LLM should respond, what kind of behaviours you can expect from it, and more
Then, you can provide a few examples in the form of "assistant"/"user" pairs
Then, you prompt the model with the true "user" message.
In this example, we'll be forgoing the 2nd step for simplicity's sake.

Utility Functions
You'll notice that we're using some utility functions from the aimakerspace module - let's take a peek at these and see what they're doing!

XYZRolePrompt
Here we have our system, user, and assistant role prompts.

Let's take a peek at what they look like:

class BasePrompt:
    def __init__(self, prompt):
        """
        Initializes the BasePrompt object with a prompt template.

        :param prompt: A string that can contain placeholders within curly braces
        """
        self.prompt = prompt
        self._pattern = re.compile(r"\{([^}]+)\}")

    def format_prompt(self, **kwargs):
        """
        Formats the prompt string using the keyword arguments provided.

        :param kwargs: The values to substitute into the prompt string
        :return: The formatted prompt string
        """
        matches = self._pattern.findall(self.prompt)
        return self.prompt.format(**{match: kwargs.get(match, "") for match in matches})

    def get_input_variables(self):
        """
        Gets the list of input variable names from the prompt string.

        :return: List of input variable names
        """
        return self._pattern.findall(self.prompt)
Then we have our RolePrompt which laser focuses us on the role pattern found in most API endpoints for LLMs.

class RolePrompt(BasePrompt):
    def __init__(self, prompt, role: str):
        """
        Initializes the RolePrompt object with a prompt template and a role.

        :param prompt: A string that can contain placeholders within curly braces
        :param role: The role for the message ('system', 'user', or 'assistant')
        """
        super().__init__(prompt)
        self.role = role

    def create_message(self, **kwargs):
        """
        Creates a message dictionary with a role and a formatted message.

        :param kwargs: The values to substitute into the prompt string
        :return: Dictionary containing the role and the formatted message
        """
        return {"role": self.role, "content": self.format_prompt(**kwargs)}
We'll look at how the SystemRolePrompt is constructed to get a better idea of how that extension works:

class SystemRolePrompt(RolePrompt):
    def __init__(self, prompt: str):
        super().__init__(prompt, "system")
That pattern is repeated for our UserRolePrompt and our AssistantRolePrompt as well.

ChatOpenAI
Next we have our model, which is converted to a format analogous to libraries like LangChain and LlamaIndex.

Let's take a peek at how that is constructed:

class ChatOpenAI:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")

    def run(self, messages, text_only: bool = True):
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        openai.api_key = self.openai_api_key
        response = openai.ChatCompletion.create(
            model=self.model_name, messages=messages
        )

        if text_only:
            return response.choices[0].message.content

        return response
❓ Question #3:
When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?

NOTE: Check out this section of the OpenAI documentation for the answer!

✅ Answer:
As per the documentation by OpenAI, You can achieve more reproducible outputs from the OpenAI API by: i. Pinning to a specific model snapshot instead of a moving alias ii. Keeping prompts and instructions consistent across calls iii.Versioning and reusing prompts rather than changing them inline iv. Using evals to detect output drift over time

Creating and Prompting OpenAI's gpt-4.1-mini!
Let's tie all these together and use it to prompt gpt-4.1-mini!

from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)

from aimakerspace.openai_utils.chatmodel import ChatOpenAI

chat_openai = ChatOpenAI()
user_prompt_template = "{content}"
user_role_prompt = UserRolePrompt(user_prompt_template)
system_prompt_template = (
    "You are an expert in {expertise}, you always answer in a kind way."
)
system_role_prompt = SystemRolePrompt(system_prompt_template)

messages = [
    system_role_prompt.create_message(expertise="Python"),
    user_role_prompt.create_message(
        content="What is the best way to write a loop?"
    ),
]

response = chat_openai.run(messages)
print(response)
Hello! The "best" way to write a loop often depends on what you're trying to accomplish, but in Python, the most common and readable way to write a loop is using a **for loop** when you have a sequence to iterate over, or a **while loop** when you want to repeat until a condition changes.

Here’s a quick overview:

### For loop (preferred for iterating over sequences)
```python
fruits = ['apple', 'banana', 'cherry']

for fruit in fruits:
    print(fruit)
```

### While loop (preferred when the number of iterations isn't known upfront)
```python
count = 0

while count < 5:
    print(count)
    count += 1
```

If you have a specific task in mind, feel free to share it, and I can help you write the most Pythonic loop for that! 😊
Task 5: Retrieval Augmented Generation
Now we can create a RAG prompt - which will help our system behave in a way that makes sense!

There is much you could do here, many tweaks and improvements to be made!

RAG_SYSTEM_TEMPLATE = """You are a helpful personal wellness assistant that answers health and wellness questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't have information about that in my wellness knowledge base"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Include a gentle reminder that users should consult healthcare professionals for medical advice when appropriate
- Only provide answers when you are confident the context supports your response."""

RAG_USER_TEMPLATE = """Context Information:
{context}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above."""

rag_system_prompt = SystemRolePrompt(
    RAG_SYSTEM_TEMPLATE,
    strict=True,
    defaults={
        "response_style": "concise",
        "response_length": "brief"
    }
)

rag_user_prompt = UserRolePrompt(
    RAG_USER_TEMPLATE,
    strict=True,
    defaults={
        "context_count": "",
        "similarity_scores": ""
    }
)
Now we can create our pipeline!

class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase, 
                 response_style: str = "detailed", include_scores: bool = False) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.response_style = response_style
        self.include_scores = include_scores

    def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs) -> dict:
        # Retrieve relevant contexts
        context_list = self.vector_db_retriever.search_by_text(user_query, k=k)
        
        context_prompt = ""
        similarity_scores = []
        
        for i, (context, score) in enumerate(context_list, 1):
            context_prompt += f"[Source {i}]: {context}\n\n"
            similarity_scores.append(f"Source {i}: {score:.3f}")
        
        # Create system message with parameters
        system_params = {
            "response_style": self.response_style,
            "response_length": system_kwargs.get("response_length", "detailed")
        }
        
        formatted_system_prompt = rag_system_prompt.create_message(**system_params)
        
        user_params = {
            "user_query": user_query,
            "context": context_prompt.strip(),
            "context_count": len(context_list),
            "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = rag_user_prompt.create_message(**user_params)

        return {
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), 
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }
rag_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai,
    response_style="detailed",
    include_scores=True
)

result = rag_pipeline.run_pipeline(
    "What are some natural remedies for improving sleep quality?",
    k=3,
    response_length="comprehensive", 
    include_warnings=True,
    confidence_required=True
)

print(f"Response: {result['response']}")
print(f"\nContext Count: {result['context_count']}")
print(f"Similarity Scores: {result['similarity_scores']}")
Response: Based on the provided context, some natural remedies for improving sleep quality include:

1. **Cognitive Behavioral Therapy for Insomnia (CBT-I):** A therapeutic approach aimed at changing sleep habits and misconceptions about sleep (Source 1).

2. **Relaxation techniques:** For example, progressive muscle relaxation, which can help ease tension (Source 1, Source 3).

3. **Herbal teas:** Such as chamomile or valerian root, which are known for their calming effects (Source 1).

4. **Magnesium supplements:** These may support better sleep, but it is important to consult a healthcare provider before use (Source 1).

5. **Meditation and mindfulness practices:** These can calm the mind and promote relaxation before bedtime (Source 1).

Additional sleep hygiene practices that support natural sleep improvement include:
- Avoiding caffeine after 2 PM
- Exercising regularly but not too close to bedtime
- Limiting alcohol and heavy meals before bed
- Maintaining a consistent sleep schedule
- Creating a relaxing bedtime routine (reading, gentle stretching, warm bath)
- Keeping the bedroom cool (65-68°F), dark, quiet, and comfortable with quality mattress and pillows (Sources 1, 2, 3)

These natural remedies and habits collectively promote better sleep quality. However, it's advisable to consult a healthcare professional for personalized medical advice, especially when considering supplements or if experiencing ongoing sleep difficulties.

Context Count: 3
Similarity Scores: ['Source 1: 0.553', 'Source 2: 0.521', 'Source 3: 0.439']
❓ Question #4:
What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?

What is that strategy called?

NOTE: You can look through our OpenAI Responses API notebook for an answer to this question if you get stuck!

✅ Answer:
As per the OpenAI Response API doc, to make the LLM produce more thoughtful and detailed responses, you can explicitly encourage reasoning in the prompt or configuration.This strategy is called Chain-of-Thought prompting (or more generally, reasoning prompting).

In the Responses API, this is supported by: i. Asking the model to think through the problem step by step implicitly ii. Increasing the model’s reasoning effort (e.g., reasoning={"effort": "medium" | "high"}) iii. Providing clear developer/system instructions that emphasize depth and care

🏗️ Activity #1:
Enhance your Personal Wellness Assistant in some way!

Suggestions are:

PDF Support: Allow it to ingest wellness PDFs (meal plans, workout guides, medical information sheets)
New Distance Metric: Implement a different similarity measure - does it improve retrieval for health queries?
Metadata Support: Add metadata like topic categories (exercise, nutrition, sleep) or difficulty levels to help filter results
Different Embedding Model: Try a different embedding model - does domain-specific tuning help for health content?
Multi-Source Ingestion: Add the capability to ingest content from YouTube health videos, podcasts, or health blogs
While these are suggestions, you should feel free to make whatever augmentations you desire! Think about what features would make this wellness assistant most useful for your personal health journey.

When you're finished making the augments to your RAG application - vibe check it against the old one - see if you can "feel the improvement"!

NOTE: These additions might require you to work within the aimakerspace library - that's expected!

NOTE: If you're not sure where to start - ask Cursor (CMD/CTRL+L) to guide you through the changes!

### YOUR CODE HERE
%pip install -q numpy

# 1. Metadata + Vector DB wrapper (NumPy)
# 2. Build your DB from your existing split chunks
# 3. Retrieval demo: before vs after filtering
# 4. Plug into your existing RAG pipeline (minimal change)

# 1. Metadata + Vector DB wrapper (NumPy)
import re
import numpy as np
from numpy.linalg import norm
import asyncio

def cosine_similarity(v1, v2, eps=1e-12):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    return float(np.dot(v1, v2) / (max(norm(v1), eps) * max(norm(v2), eps)))

def simple_topic_classifier(text: str) -> str:
    t = text.lower()

    topic_keywords = {
        "sleep": ["sleep", "insomnia", "bedtime", "melatonin", "nap", "circadian", "snore"],
        "nutrition": ["nutrition", "diet", "protein", "carb", "fat", "fiber", "vitamin", "calorie", "meal", "hydration", "water"],
        "exercise": ["exercise", "workout", "strength", "cardio", "stretch", "yoga", "mobility", "run", "walk", "lift"],
        "stress": ["stress", "anxiety", "mindfulness", "meditation", "breathing", "cortisol", "relax"],
        "habits": ["habit", "routine", "consistency", "goal", "journal", "track", "behavior"],
    }

    scores = {k: 0 for k in topic_keywords}
    for topic, kws in topic_keywords.items():
        for kw in kws:
            if kw in t:
                scores[topic] += 1

    best_topic = max(scores, key=scores.get)
    return best_topic if scores[best_topic] > 0 else "general"


class MetadataVectorDB:
    """
    Stores: [{"text": str, "embedding": np.array, "metadata": dict}, ...]
    Supports:
      - topic filter
      - top-k cosine
      - optional MMR for diverse contexts
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.items = []

    async def abuild_from_texts(self, texts, source_name="HealthWellnessGuide.txt"):
        # Batch embed (async) for speed
        embeddings = await self.embedding_model.async_get_embeddings(texts)

        for i, (txt, emb) in enumerate(zip(texts, embeddings)):
            meta = {
                "source": source_name,
                "chunk_id": i,
                "topic": simple_topic_classifier(txt),
                "length": len(txt),
            }
            self.items.append({
                "text": txt,
                "embedding": np.asarray(emb, dtype=np.float32),
                "metadata": meta
            })
        return self

    async def asearch(self, query: str, k=4, topic_filter=None, use_mmr=True, mmr_lambda=0.7, candidate_pool=30):
        # Embed query
        q_emb = await self.embedding_model.async_get_embeddings([query])
        q_vec = np.asarray(q_emb[0], dtype=np.float32)

        # Filter candidates by topic (if provided)
        candidates = self.items
        if topic_filter:
            tf = set(topic_filter if isinstance(topic_filter, list) else [topic_filter])
            candidates = [it for it in self.items if it["metadata"]["topic"] in tf]

        if not candidates:
            return []

        # Score all candidates by cosine similarity
        scored = []
        for it in candidates:
            s = cosine_similarity(q_vec, it["embedding"])
            scored.append((it, s))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Take a candidate pool for MMR (speed + quality)
        pool = scored[:min(candidate_pool, len(scored))]

        if not use_mmr:
            return [(it["text"], score, it["metadata"]) for it, score in pool[:k]]

        # MMR: pick items that are relevant AND diverse
        selected = []
        selected_vecs = []

        while len(selected) < min(k, len(pool)):
            best = None
            best_mmr = -1e9

            for it, rel in pool:
                if any(it is s_it for s_it, _, _ in selected):
                    continue

                if not selected_vecs:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = max(cosine_similarity(it["embedding"], sv) for sv in selected_vecs)

                mmr_score = mmr_lambda * rel - (1 - mmr_lambda) * diversity_penalty

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best = (it, rel)

            if best is None:
                break

            it, rel = best
            selected.append((it, rel, it["metadata"]))
            selected_vecs.append(it["embedding"])

        return [(it["text"], rel, meta) for it, rel, meta in selected]

#2. Build your DB from your existing split chunks
from aimakerspace.openai_utils.embedding import EmbeddingModel

embedding_model = EmbeddingModel()

meta_db = MetadataVectorDB(embedding_model)
meta_db = asyncio.run(meta_db.abuild_from_texts(split_documents, source_name="HealthWellnessGuide.txt"))

# quick sanity check: topic distribution
from collections import Counter
Counter([it["metadata"]["topic"] for it in meta_db.items])

#3. Retrieval demo: before vs after filtering
#Normal search (no topic filter)
query = "What are some natural remedies for improving sleep quality?"
results = asyncio.run(meta_db.asearch(query, k=4, topic_filter=None, use_mmr=True))

for i, (text, score, meta) in enumerate(results, 1):
    print(f"\n--- Source {i} | score={score:.3f} | topic={meta['topic']} | chunk={meta['chunk_id']} ---")
    print(text[:500])

# Filtered search (sleep only) — this is the enhancement
results_sleep = asyncio.run(meta_db.asearch(query, k=4, topic_filter="sleep", use_mmr=True))

for i, (text, score, meta) in enumerate(results_sleep, 1):
    print(f"\n--- Source {i} | score={score:.3f} | topic={meta['topic']} | chunk={meta['chunk_id']} ---")
    print(text[:500])

#4. Plug into your existing RAG pipeline (minimal change)
# OLD:
# context_list = self.vector_db_retriever.search_by_text(user_query, k=k)

# NEW:
user_query = "What are some natural remedies for improving sleep quality?"
k = 3
topic_filter = "sleep"

context_list = asyncio.run(
    meta_db.asearch(
        user_query,
        k=k,
        topic_filter=topic_filter,
        use_mmr=True
    )
)
context_prompt = ""
similarity_scores = []

for i, (context, score, meta) in enumerate(context_list, 1):
    context_prompt += f"[Source {i} | topic={meta['topic']} | chunk={meta['chunk_id']}]: {context}\n\n"
    similarity_scores.append(f"Source {i}: {score:.3f} ({meta['topic']})")

result = rag_pipeline.run_pipeline(
    "What are some natural remedies for improving sleep quality?",
    k=3,
    response_length="comprehensive",
    topic_filter="sleep"   # ✅ new capability
)
Note: you may need to restart the kernel to use updated packages.

--- Source 1 | score=0.553 | topic=sleep | chunk=8 ---
 Avoid caffeine after 2 PM
- Exercise regularly, but not too close to bedtime
- Limit alcohol and heavy meals before bed

Creating an optimal sleep environment:
- Temperature: 65-68 degrees Fahrenheit (18-20 Celsius)
- Darkness: Use blackout curtains or a sleep mask
- Quiet: Consider white noise machines or earplugs
- Comfort: Invest in a quality mattress and pillows

Chapter 9: Understanding and Managing Insomnia

Insomnia is difficulty falling asleep, staying asleep, or waking too early. It af

--- Source 2 | score=0.522 | topic=sleep | chunk=7 ---
t. Sleep occurs in cycles of about 90 minutes, alternating between REM (rapid eye movement) and non-REM sleep.

The four stages of sleep:
- Stage 1: Light sleep, easy to wake, lasts 5-10 minutes
- Stage 2: Body temperature drops, heart rate slows, prepares for deep sleep
- Stage 3: Deep sleep, difficult to wake, body repairs and regenerates
- REM Sleep: Brain is active, dreams occur, important for memory and learning

Chapter 8: Improving Sleep Quality

Sleep hygiene refers to habits and practic

--- Source 3 | score=0.438 | topic=stress | chunk=19 ---
 media feeds for positivity

APPENDIX: QUICK REFERENCE GUIDES

Emergency Stress Relief (5 minutes or less):
1. Box breathing: 4 counts in, 4 hold, 4 out, 4 hold
2. Cold water on wrists and face
3. Step outside for fresh air
4. Progressive muscle relaxation
5. Name 5 things you can see, 4 hear, 3 feel, 2 smell, 1 taste

Quick Energy Boosters:
1. 10 jumping jacks
2. Glass of cold water
3. Step outside for sunlight
4. Power pose for 2 minutes
5. Healthy snack (nuts, fruit)
6. Brief walk around the 

--- Source 4 | score=0.438 | topic=habits | chunk=13 ---
ke at a consistent time
- Drink a glass of water immediately
- 5-10 minutes of stretching or light movement
- Healthy breakfast
- Brief mindfulness practice or journaling
- Review goals and priorities for the day

Avoid these morning habits:
- Checking phone immediately upon waking
- Skipping breakfast
- Hitting snooze multiple times
- Rushing without preparation
- Starting with negative news

Chapter 15: Evening Wind-Down Routines

A consistent evening routine signals to your body that it's tim

--- Source 1 | score=0.553 | topic=sleep | chunk=8 ---
 Avoid caffeine after 2 PM
- Exercise regularly, but not too close to bedtime
- Limit alcohol and heavy meals before bed

Creating an optimal sleep environment:
- Temperature: 65-68 degrees Fahrenheit (18-20 Celsius)
- Darkness: Use blackout curtains or a sleep mask
- Quiet: Consider white noise machines or earplugs
- Comfort: Invest in a quality mattress and pillows

Chapter 9: Understanding and Managing Insomnia

Insomnia is difficulty falling asleep, staying asleep, or waking too early. It af

--- Source 2 | score=0.521 | topic=sleep | chunk=7 ---
t. Sleep occurs in cycles of about 90 minutes, alternating between REM (rapid eye movement) and non-REM sleep.

The four stages of sleep:
- Stage 1: Light sleep, easy to wake, lasts 5-10 minutes
- Stage 2: Body temperature drops, heart rate slows, prepares for deep sleep
- Stage 3: Deep sleep, difficult to wake, body repairs and regenerates
- REM Sleep: Brain is active, dreams occur, important for memory and learning

Chapter 8: Improving Sleep Quality

Sleep hygiene refers to habits and practic

--- Source 3 | score=0.334 | topic=sleep | chunk=20 ---
cklist:
- 8 glasses of water
- 5+ servings of fruits/vegetables
- 30 minutes of movement
- 7-9 hours of sleep
- Moment of mindfulness
- Connection with someone you care about
- Time for something you enjoy
