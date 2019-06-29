preprocess_topics_amazon.ipynb - creating and preprocessing the dataset

pre_training2.ipynb - pre-training the user and item embeddings (only item embeddings are used in the model)

Reviews_CNN_Prep_lowram.ipynb - preprocessing the desired dataset for the CNN model

Reviews_CNN.ipynb - CNN model to learn topic representations (requires also the results from LDA)

Model2-topics_daria.py - training and testing the model

	python3 Model2-topics.py first_k self_attention_w LR dropout batch_size features item_emb user_emb rnn_dim 
	
sarah/daria run_num
	
	python3 Model2-topics.py 10 16 0.01 0.5 8192 50 16 32 70 sarah 1
	python3 Model2-topics.py 10 16 0.01 0.5 8192 50 16 32 16 daria 1 (rnn_dim == item_emb)
