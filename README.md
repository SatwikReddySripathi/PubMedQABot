# PubMedQA_BOT
This project implements a medical question-answering chatbot leveraging a Fine-Tuned TinyLlama model and a *Retrieval-Augmented Generation (RAG) framework. Built using the PubMedQA dataset, the system combines advanced retrieval mechanisms powered by FAISS and context-aware generation to deliver accurate and contextually grounded answers in the biomedical domain. The chatbot features a user-friendly interface developed with ChainLIT, ensuring seamless interaction for researchers, medical professionals, and students. The project showcases enhanced performance through fine-tuning techniques like PEFT LoRA and integrates state-of-the-art embeddings from Hugging Face for precise query handling and document retrieval. This repository includes code for model fine-tuning, retrieval integration, and UI deployment, along with evaluation metrics for performance benchmarking.


## 📚 Here are some of the features of the tinyllama-finetune-Medical-Chatbot:

 - It uses the DhanushAkula/Llama-2-7b-chat-finetune, which is a **large language model (LLM)** that has been fine-tuned.
   * Name - **tinyllama-finetune**
   * Quant method - q2_K
   * Bits - 2
   * Size - **2.5 GB**
   * Max RAM required - 5.37 GB
 
   * **Model:** Know more about model **[Llama-2-7b-chat-finetune](https://huggingface.co/DhanushAkula/tinyllama-finetune)**
 - It is trained on the pdf **[PubMedQA Dataset](https://github.com/pubmedqa/pubmedqa)**, which is a comprehensive medical reference that provides information on a wide range of medical topics. This means that the chatbot is able to answer questions about a variety of medical topics.
 - This is a sophisticated medical chatbot, developed using TinyLlama and Sentence Transformers. Powered by **[Langchain](https://python.langchain.com/docs/get_started/introduction)** and **[Chainlit](https://docs.chainlit.io/overview)**, This bot operates on a powerful CPU computer that boasts a minimum of
    * Operating system: Linux, macOS, or Windows
    * CPU: Intel® Core™ i3
    * RAM: **8 GB**
    * Disk space: 7 GB
    * GPU: None **(CPU only)**


## 🚀 Setup
1. Open Git Bash.
2. Change the current working directory to the location where you want the cloned directory.
3. Type `git clone`, and then paste the URL you copied earlier.
```bash
   git clone https://github.com/SatwikReddySripathi/PubMedQABot
```
Press Enter to create your local clone.
4. Install the pip packages in requirements.txt
 ```bash
   pip install -r requirements.txt
 ```
5. Now run it!
```terminal
   chainlit run model.py -w
```
## 📖 ChatBot Conversession
### ⛓️Chainlit ver. on [#v1.0.1.dev20230913](https://github.com/ThisIs-Developer/Llama-2-GGML-Medical-Chatbot/releases/tag/v1.0.1.dev20230913)
![ChatBot Conversation img-1]()
![ChatBot Conversation img-2]()



## Contributors
@satwikReddySripathi
@DhanushAkula
## 📄 License

This project is licensed under the [MIT License](https://github.com/ThisIs-Developer/Llama-2-GGML-Medical-Chatbot/blob/main/LICENSE).
