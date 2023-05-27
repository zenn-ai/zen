conda create -n py310 python=3.10 -y && \
    source activate py310 && \
    pip install accelerate==0.18.0 \
        appdirs==1.4.4 \
        bitsandbytes==0.37.2 \
        datasets==2.10.1 \
        fire==0.5.0 \
        torch==2.0.0 \
        sentencepiece==0.1.97 \
        tensorboardX==2.6 \
        gradio==3.23.0 \
        seaborn==0.12.2 \
        ipykernel \
        ipywidgets \
        tensorboard==2.12.2 \
        langchain==0.0.154 \
        sentence_transformers==2.2.2 \
        unstructured==0.6.2 \
        deepspeed==0.9.2 && \
    pip install flash_attn && \
    pip install git+https://github.com/huggingface/peft.git && \
    pip install git+https://github.com/huggingface/transformers.git && \
    python -m ipykernel install --user --name py310 --display-name "Python (3.10)"
