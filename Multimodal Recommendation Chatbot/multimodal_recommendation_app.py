import streamlit as st

import os

GOOGLE_API_KEY = "AIzaSyCdoRgZp_qChjIzZt4NYFyN-L7ZCtYsIgQ"  # add your GOOGLE API key here
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

import pandas as pd

df_styles = pd.read_csv('E:\AIMT\Chatbot\styles.csv')

df_styles.head(5)

from llama_index.core.schema import TextNode

productDisplayName = df_styles['productDisplayName'].tolist()
nodes = []
for i in range(len(productDisplayName)):
    meta_data = {
                "gender" : df_styles['gender'][i],
                "subCategory" : df_styles['subCategory'][i],
                "articleType" : df_styles['articleType'][i],
                "baseColour" : df_styles['baseColour'][i],
                "season" : df_styles['season'][i],
                "usage" : df_styles['usage'][i],
                }

    nodes.append(TextNode(text=productDisplayName[i], metadata=meta_data))

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.core import StorageContext
import qdrant_client
from qdrant_client import QdrantClient



#Create a local Qdrant vector store
#client = qdrant_client.QdrantClient(path="qdrant_gemini_10")
client = QdrantClient(url="http://localhost:6333")

vector_store = QdrantVectorStore(client=client, collection_name="text_collection")

# Using the embedding model to Gemini
Settings.embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY
)
Settings.llm = Gemini(model_name="models/gemini-1.5-pro", api_key=GOOGLE_API_KEY)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

query_engine = index.as_query_engine(
    #similarity_top_k=1,
)

response = query_engine.query(
    "recommend a black men shirt for summer"
)

image_metadata_dict = {}

df_images = pd.read_csv('images.csv')
image_files = df_images['filename'].tolist()
image_files = [f"images/{image_files[i]}" for i in range(len(image_files))]
image_urls = df_images['link'].tolist()

for image_file, image_url in zip(image_files, image_urls):
    image_filename = os.path.basename(image_file)
    image_file_path = os.path.abspath(image_file)
    image_metadata_dict[image_filename] = {
                                          "filename": image_filename,
                                          "img_path": image_file_path,
                                          "url": image_url,
                                          }

import matplotlib.pyplot as plt
from PIL import Image
import os
def plot_images(image_metadata_dict):
    images = []
    images_shown = 0
    for image_filename in image_metadata_dict:
        img_path = image_metadata_dict[image_filename]["img_path"]
        if os.path.isfile(img_path):
            # open the image file and convert it to RGB colorspace.
            filename = image_metadata_dict[image_filename]["filename"]
            try:
                image = Image.open(img_path).convert("RGB")
                # plot the image in a subplot of an 8x8 grid, also disables the tick labels on the axes to make the plot cleaner.
                plt.subplot(8, 8, len(images) + 1)
                plt.imshow(image)
                plt.xticks([])
                plt.yticks([])

                images.append(filename)
                images_shown += 1
                if images_shown >= 64:
                    break
            except Exception as e:
                print(f"Error opening image {img_path}: {e}")
        else:
            print(f"File {img_path} does not exist.")

    plt.tight_layout()
    plt.show()


import clip
import numpy as np

model, preprocess = clip.load("ViT-B/32")
device = "cuda"
model = model.to(device)
input_resolution = model.visual.input_resolution


context_length = model.context_length
vocab_size = model.vocab_size


import torch

img_emb_dict = {}
with torch.no_grad():
    for image_filename in image_metadata_dict:
        img_file_path = image_metadata_dict[image_filename]["img_path"]
        if os.path.isfile(img_file_path):
            try:
                image = (
                    preprocess(Image.open(img_file_path)).unsqueeze(0).to(device)
                )
                image_features = model.encode_image(image)
                img_emb_dict[image_filename] = image_features
            except:
                pass

from llama_index.core import Document

img_documents = []
for image_filename in image_metadata_dict:
    if image_filename in img_emb_dict:
        filename = image_metadata_dict[image_filename]["filename"]
        filepath = image_metadata_dict[image_filename]["img_path"]
        url = image_metadata_dict[image_filename]["url"]

        newImgDoc = Document(
                                text=filename,
                                metadata={
                                        "filepath": filepath,
                                        "url": url
                                        })

        newImgDoc.embedding = img_emb_dict[image_filename].tolist()[0]
        img_documents.append(newImgDoc)

text_client = QdrantClient(url="http://localhost:6333")
image_vector_store = QdrantVectorStore(
                                    client=text_client,
                                    collection_name="image_collection"
                                    )

storage_context = StorageContext.from_defaults(vector_store=image_vector_store)
product_image_index = VectorStoreIndex.from_documents(
                                            img_documents,
                                            storage_context=storage_context
                                            )

from llama_index.core.vector_stores import VectorStoreQuery
def retrieve_results_from_image_index(query):
    text = clip.tokenize(query).to(device)
    query_embedding = model.encode_text(text).tolist()[0]
    image_vector_store_query = VectorStoreQuery(
                                                query_embedding=query_embedding,
                                                similarity_top_k=1, # only return 1 image
                                                mode="default",
                                                )

    image_retrieval_results = image_vector_store.query(image_vector_store_query)
    return image_retrieval_results

# Function to convert image path to URL
def get_local_image_url(image_path):
    base_url = "http://localhost:5000/images/"
    image_filename = os.path.basename(image_path)
    return f"{base_url}{image_filename}"

import replicate
os.environ["REPLICATE_API_TOKEN"] = "r8_V6c7mtVNEyBne2E2x9Q5HHhNnpa0IHt0pyCQj"

def plot_image_retrieve_results(image_retrieval_results):
    plt.figure(figsize=(16, 5))

    img_cnt = 0
    for returned_image, score in zip(
        image_retrieval_results.nodes, image_retrieval_results.similarities
    ):
        img_path = returned_image.metadata["filepath"]
        image_url = returned_image.metadata["url"]
        image = Image.open(img_path).convert("RGB")

        plt.subplot(2, 3, img_cnt + 1)
        plt.title("{:.4f}".format(score))

        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        img_cnt += 1

        return image_url, img_path

def llava_inference(image, prompt, max_tokens=100, temperature=0.5):
    client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
    output = client.run(
        "yorickvp/llava-13b:c293ca6d551ce5e74893ab153c61380f5bcbd80e02d49e08c582de184a8f6c83",
        input={
            "image": image,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
    )
    return output

import textwrap

# Function to perform multimodal retrieval and inference
def multimodal_retrieval(query, prompt="""
                        Consider below context to provide your suggestions on provided image's suitability.

                        {context}
                        """):
    image_retrieval_results = retrieve_results_from_image_index(query)
    image_url, img_path = plot_image_retrieve_results(image_retrieval_results)
    img_path = get_local_image_url(img_path)
    context = str(query_engine.query(query))

    output = llava_inference(image_url, prompt.format(context=context))
    
    # Print the type and content of the output for debugging
    #print(f"Type of output: {type(output)}")
    #print(f"Output content: {output}")
    
    return output, image_url, img_path

# Streamlit application
def main():
    st.title("Multimodal Recommendation Bot")
    
    query = st.text_input("Enter your query:")
    
    if st.button("Submit"):
        if query:
            result, image_url, img_path = multimodal_retrieval(query)
            st.write("Recommendation Result:")
            st.write(textwrap.fill(result, width=100))
            st.write("Product Link:")
            st.write(textwrap.fill(image_url, width=100))
            #st.write("Product Image:")
            #st.write(textwrap.fill(img_path, width=100))
            st.image(img_path, caption='Product Image', use_column_width=True)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()