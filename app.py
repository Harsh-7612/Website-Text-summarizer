import validators
import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## streamlit APP
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website")
st.title("Langchain: Summarize text from YT or Website")
st.subheader("Summarize URL")


# Get the Groq API Key and URL field to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key", value="", type="password")

generic_url=st.text_input("URL",label_visibility="collapsed") 

#Model initialisation
llm=ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt=PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip(): #strip removes empty characters
        st.error("Please provide the information")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can maybe a YT video url or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                ##loading the website data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(url=[generic_url],ssl_verify=False)
                docs=loader.load()
                ##chain for summarisation
                chain=load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)                    
                
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")        