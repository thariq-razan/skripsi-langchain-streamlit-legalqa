# SET UP
import streamlit as st
import pinecone
from streamlit_chat import message
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
os.environ["OPENAI_API_KEY"] = st.secrets.openai_credentials.api_key
pinecone.init(api_key=st.secrets.pinecone_credentials.api_key, environment=st.secrets.pinecone_credentials.environment)


# BACKEND
def chain_call():
    docsearch = Pinecone.from_existing_index(index_name="langchainlegalpdf", embedding=OpenAIEmbeddings(), namespace="skripsi_4_peraturan")
    prompt_template = """
    PERINTAH
    Kamu adalah Kecerdasan Artifisial yang diprogram untuk merespon berbagai pertanyaan dari pengguna yang terkait dengan domain hukum di Indonesia.
    Kamu akan diberikan konteks yang harus dijadikan sebagai sumber pengetahuan utama dalam merespon pertanyaan pengguna.
    ---
    CONTOH 1
    Perhatikan contoh prompt di bawah ini.
    "Siapa yang termasuk sebagai anak?"
    Perhatikan contoh konteks di bawah ini.
    "[Berikut adalah isi Pasal 1 angka 26 pada UU 13/2003] Anak adalah setiap orang yang berumur dibawah 18 (delapan belas) tahun.",
    "[Berikut adalah isi Pasal 73 pada UNDANG-UNDANG REPUBLIK INDONESIA NOMOR 13 TAHUN 2003 TENTANG KETENAGAKERJAAN] Anak dianggap bekerja bilamana berada di tempat kerja, kecuali dapat dibuktikan sebaliknya."
    Prompt yang dicontohkan merupakan pertanyaan yang terkait dengan domain hukum di Indonesia, alhasil kamu dapat memberikan respons seperti di bawah ini.
    Anak adalah setiap orang yang berumur dibawah 18 (delapan belas) tahun. Sumber: Pasal 1 angka 26 UU 13/2003
    CONTOH 2
    Perhatikan contoh prompt di bawah ini.
    "Di UU No. 1 Tahun 2001, apa isi Pasal 31?"
    Perhatikan contoh konteks di bawah ini.
    "[Berikut 10 buah pasal dari UU Nomor 1 Tahun 2001 yang telah dihapus pada UU Nomor 2 Tahun 2002] Pasal 12, Pasal 21, Pasal 24, Pasal 28
    Pasal 30, Pasal 31, Pasal 100, Pasal 101, Pasal 102, Pasal 150"
    Prompt yang dicontohkan di atas merupakan pertanyaan yang terkait dengan domain hukum di Indonesia, alhasil kamu dapat memberikan respons seperti di bawah ini.
    Pasal 31 di UU No. 1 Tahun 2001 telah dihapus oleh UU Nomor 2 Tahun 2002. Sumber: UU Nomor 2 Tahun 2002
    CONTOH 3
    Berikut contoh-contoh prompt lain terkait domain hukum di Indonesia yang mungkin diberikan oleh pengguna.
    "Yg disebut sbg anak adalah siapa?"
    "Apa fungsi dari lembaga kerja sama bipartit?"
    "Apa sanksi bagi pelanggaran hak mogok kerja pekerja?"
    "Buatkan draf perjanjian magang untuk saya"
    ---
    Jawablah pertanyaan di bawah ini.
    Pertanyaan: {question}
    ---
    Konteks: {context}
    ---
    ALTERNATIF
    Jika konteks yang diberikan tidak dapat digunakan untuk menjawab pertanyaan pengguna,
    abaikan segala hal yang telah diinstruksikan sebelumnya dan kembalikan respons di bawah ini:
    Maaf, saya tidak bisa merespon prompt tersebut.
    """
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])},
                                 return_source_documents=True
                                 )
    return qa


# FRONTEND
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

st.set_page_config(page_title="Perpy")
st.title(":red[Perpy] (beta)", anchor=False)
st.subheader("An :red[Indonesian Legal] QA System Powered by OpenAI", anchor=False)
st.write("""
Halo! Saya telah disuplai oleh sumber data eksternal berisi 
Peraturan Perundang-undangan Republik Indonesia pada klaster Ketenagakerjaan. 
Apakah Anda memiliki pertanyaan yang bisa saya bantu?
""")

def submit():
    chain = chain_call()
    output = chain({"query": st.session_state.input})
    st.session_state.past.append(st.session_state.input)
    st.session_state.generated.append(output["result"])
    st.session_state.input = ""
    print("\n-----retrieved context-----")
    print(*output["source_documents"], sep='\n-----\n')
    
input_widget = st.text_input(
    label="Masukkan pertanyaan mu di bawah ini",
    label_visibility="collapsed",
    placeholder="Apa definisi dari ketenagakerjaan?",
    key="input",
    on_change=submit # WARNING: widget akan disubmit apabila setelah memasukkan nilai, kursor pengguna berpindah tempat (karena sudah terhitung on change)
)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")