import gradio as gr
import pandas as pd
from random import shuffle
from faiss import IndexFlatIP
from llama_index.core.evaluation.retrieval.metrics import HitRate, MRR

from Retriver.vectorSearch import VectorSearch
from Retriver.bm25Search import bm25Search
from config import dimension, gradio_host, gradio_port
from preprocess.llamainex_add_corpus import queries, query_relevant_docs

retrieve_methods = ["embedding", "bm25"]


def get_metric(search_query, search_result):
    hit_rate = HitRate().compute(query=search_query,
                                 expected_ids=query_relevant_docs[search_query],
                                 retrieved_ids=[_.id_ for _ in search_result])

    mrr = MRR().compute(query=search_query,
                        expected_ids=query_relevant_docs[search_query],
                        retrieved_ids=[_.id_ for _ in search_result])
    return [hit_rate.score, mrr.score]

def get_retrieve_result(retriever_list, retrieve_top_k, query_textbox, query_dropdown, usage):
    if usage == 'Online':
        columns = {"top_k": [f"top_{k + 1}" for k in range(retrieve_top_k)]}
        retrieve_query = query_textbox
    else:
        columns = {"metric_&_top_k": ["Hit Rate", "MRR"] + [f"top_{k + 1}" for k in range(retrieve_top_k)]}
        retrieve_query = query_dropdown

    if "bm25" in retriever_list:
        bm25_retriever = bm25Search(top_k=retrieve_top_k)
        search_result = bm25_retriever.retrieve(retrieve_query)
        columns["bm25"] = []
        if usage == 'Evaluation':
            columns["bm25"].extend(get_metric(retrieve_query, search_result))
        for i, node in enumerate(search_result, start=1):
            columns["bm25"].append(node.text)

    if "embedding" in retriever_list:
        faiss_index = IndexFlatIP(dimension)
        vector_search_retriever = VectorSearch(top_k=retrieve_top_k, faiss_index=faiss_index)
        search_result = vector_search_retriever.retrieve(str_or_query_bundle=retrieve_query)
        columns["embedding"] = []
        if usage == 'Evaluation':
            columns["embedding"].extend(get_metric(retrieve_query, search_result))
        for i in range(retrieve_top_k):
            columns["embedding"].append(search_result[i].text)
        faiss_index.reset()
    retrieve_df = pd.DataFrame(columns)
    return retrieve_df

def update_query_component(usage, queries):
    if usage == "Online":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        shuffle(queries)
        return gr.update(visible=False), gr.update(visible=True, choices=queries, value=queries[0])

with gr.Blocks() as demo:
    usage = gr.Radio(choices=["Online", "Evaluation"], label="Usage", value="Online")
    retrievers = gr.CheckboxGroup(choices=retrieve_methods,
                                  type="value",
                                  label="Retrieve Methods")

    top_k = gr.Dropdown(list(range(1, 10)), label="top_k", value=3)

    query_textbox = gr.Textbox(lines=1, placeholder="Please input your query?", visible=True)
    query_dropdown = gr.Dropdown(choices=queries, label="query", visible=False)

    queries_state = gr.State(queries)
    usage.change(fn=update_query_component, inputs=[usage, queries_state], outputs=[query_textbox, query_dropdown])


    # 设置输出组件
    result_table = gr.DataFrame(label='Result', wrap=True)
    theme = gr.themes.Base()
    # 设置按钮
    submit = gr.Button("Submit")
    submit.click(fn=get_retrieve_result,
                 inputs=[retrievers, top_k, query_textbox, query_dropdown, usage],
                 outputs=result_table)


demo.launch(server_port=gradio_port, server_name=gradio_host)