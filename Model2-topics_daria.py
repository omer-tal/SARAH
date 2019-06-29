#!/usr/bin/env python
# coding: utf-8

import numpy as np
from time import time
from sklearn.metrics import accuracy_score,roc_auc_score,mean_squared_error
import tensorflow as tf
import pickle
import operator
import random
import heapq
import math
import sys

def load_file(file):
    users = []
    pois = []
    scores = []
    historic_pois = []
    reviews = []
    for line in file:
        record = line.split("\t")
        user_id = int(record[0])
        poi_id = int(record[1])
        score = float(record[2])
        last_k = [int(s) for s in record[4].split(',')]
        review = record[5]
        users.append(user_id)
        pois.append(poi_id)
        scores.append(score)
        historic_pois.append(last_k)
        reviews.append(review)
    return users,pois,scores,historic_pois,reviews

# command syntax: python3 Model2-topics.py first_k self_attention_w LR dropout batch_size features item_emb user_emb rnn_dim sarah/daria run_num
# example for sarah: python3 Model2-topics.py 10 16 0.01 0.5 8192 50 16 32 70 sarah 1
# example for daria: python3 Model2-topics.py 10 16 0.01 0.5 8192 50 16 32 16 daria 1 (rnn_dim == item_emb)

# Dataset stats
AMAZON_TYPE = "amazon_movies"
PREFIX = "879_amazon_"
DIR = AMAZON_TYPE + "/" + PREFIX 
WEIGHTS_DIR = "result/" + DIR

if len(sys.argv)>6:
    NUM_FEATURES = int(sys.argv[6])
else:
    NUM_FEATURES = 50#25#661#133
print("num features = {}".format(NUM_FEATURES))
FEATURES_PER_POI = NUM_FEATURES#50#25
FEATURE_TYPE = 'topics'
LDA_TYPE = ''#'_plusplus'#''
FEATURE_EMB_FILE = "data_cnn/" + AMAZON_TYPE + "/topic_embeddings_" + str(NUM_FEATURES) +LDA_TYPE+ ".pkl"
RESTORE = False

# Load poi-categories dictionary
with open(DIR + 'topics' + LDA_TYPE + "_" + str(NUM_FEATURES) + '.pkl', 'rb') as f:
    poi_categories = pickle.load(f)

# Flatten all categoires into one dimensional set
categories_list = set()
for categories in poi_categories.values():
    for category in categories:
        categories_list.add(category)
    
if len(sys.argv)>1:
    FIRST_K = int(sys.argv[1])
else:
    FIRST_K = 8
print("using {} K values for user".format(FIRST_K))

DATA_DIR = DIR + str(FIRST_K) + "_"
    
# Load training data
with open(DATA_DIR + "train.txt") as train_file:
    train_users, train_pois, train_scores, train_k_pois, train_reviews  = load_file(train_file)

TRAIN_SIZE = len(train_users)
print("Train size = {}".format(TRAIN_SIZE))
    
# Load categories of training pois
train_features = []
for poi in train_pois:
    train_features.append(poi_categories[poi])
    #tmp = np.array(poi_categories[poi])
    #tmp[tmp>0] = 1
    #train_features_mask.append(tmp)
    
# Set training dataset
train_set = {'users': train_users, 'pois': train_pois, 'scores': train_scores, 'k_pois': train_k_pois, 'reviews': train_reviews, 
             'features': train_features}

# Load test data
with open(DATA_DIR + "test.txt") as test_file:
    test_users, test_pois, test_scores, test_k_pois, test_reviews = load_file(test_file)
# Load categories of test pois
test_features = []
for poi in test_pois:
    test_features.append(poi_categories[poi])
    #tmp = np.array(poi_categories[poi])
    #tmp[tmp>0] = 1
    #test_features_mask.append(tmp)
    
# Set test dataset
test_set = {'users': test_users, 'pois': test_pois, 'scores': test_scores, 'k_pois': test_k_pois, 'reviews': test_reviews, 
            'features': test_features}

NUM_USERS = max(train_users)+1
NUM_ITEMS = max(train_pois)+1
TEST_SIZE = len(test_users)

# Check training data
assert len(train_users)==TRAIN_SIZE, "mismatch in training set size {}!={}".format(len(train_users),TRAIN_SIZE)
assert len(train_k_pois[0])==FIRST_K==len(test_k_pois[0]), "mismatch in number of positive items {}!={}!={}".format(len(train_k_pois[0]),FIRST_K,len(test_k_pois[0]))
assert max(train_users)==(NUM_USERS-1)==max(test_users), "mimatch in number of users {}!={}!={}".format(max(train_users),(NUM_USERS-1),max(test_users))
assert max(train_pois)==(NUM_ITEMS-1)==max(test_pois), "mismatch in number of item {}!={}!={}".format(max(train_pois),(NUM_ITEMS-1),max(test_pois))
assert len(test_users)==TEST_SIZE, "mismatch in test set size {}!={}".format(len(test_users),TRAIN_SIZE)


# # Model hyperparameters

# In[21]:

# Location embedding sizes
if len(sys.argv)>7:
    ITEM_EMBEDDING_SIZE = int(sys.argv[7])
else:
    ITEM_EMBEDDING_SIZE = 32#128

# User embedding sizes
if len(sys.argv)>8:
    USER_EMBEDDING_SIZE = int(sys.argv[8])
else:
    USER_EMBEDDING_SIZE = 32
    
print("item and user embedding sizes: {}, {}".format(ITEM_EMBEDDING_SIZE,USER_EMBEDDING_SIZE))

USER_FINAL_EMBEDDING = 0#32

FEATURE_EMBEDDING_SIZE = 50#128
TRAIN_FEATURE_EMBEDDING = True

if len(sys.argv)>2:
    self_attention_w = int(sys.argv[2])
else:
    self_attention_w = 16
print("self attention weights {}".format(self_attention_w))
# Attention weights
ITEM_SELF_ATTENTION_W = self_attention_w#64
USER_SELF_ATTENTION_W = self_attention_w#64
#USER_ITEM_W = 16#32

# Dropout ratios (1=no dropout)
if len(sys.argv)>4:
    DROPOUT_ITEM = float(sys.argv[4])
else:
    DROPOUT_ITEM = 0.5
print("dropout item {}".format(DROPOUT_ITEM))
DROPOUT_RNN_USER = 1#0.5

# Training parameters
if len(sys.argv)>3:
    LR = float(sys.argv[3])
else:
    LR = 0.1
print("LR={}".format(LR))
EPOCHS = 100
MIN_EPOCH_TO_SAVE = 5
ROC_DIFF_TO_SAVE = 1.002
EARLY_STOP_INTERVAL = 20
if len(sys.argv)>5:
    BATCH_SIZE = int(sys.argv[5])
else:
    BATCH_SIZE = 8192
print("batch size {}".format(BATCH_SIZE))
PRETRAIN_FILE = WEIGHTS_DIR + "pre_trained" + str(ITEM_EMBEDDING_SIZE) + "_" + str(USER_EMBEDDING_SIZE) +".ckpt"
WITH_PAD = False
if len(sys.argv)>10:
    MODEL_TYPE = sys.argv[10]
else:
    MODEL_TYPE = 'sarah'
if len(sys.argv)>9 and int(sys.argv[9])>0:# and MODEL_TYPE == 'sarah':
    USE_RNN = True
    ITEM_FINAL_EMBEDDING = int(sys.argv[9])
else:
    USE_RNN = False
    ITEM_FINAL_EMBEDDING = ITEM_EMBEDDING_SIZE
RNN_ITEM_EMBEDDING_SIZE = ITEM_FINAL_EMBEDDING#ITEM_EMBEDDING_SIZE#100
print("using rnn? {} with {} units".format(USE_RNN,ITEM_FINAL_EMBEDDING))
TOP_ITEMS = FIRST_K#4
print("r={}".format(TOP_ITEMS))
#TOP_FEATURES = 8

tf.reset_default_graph()
# # Set inputs and initial embeddings
in_item = tf.placeholder("int32",[None],'in_item')
in_item_features = tf.placeholder("int32",[None,FEATURES_PER_POI],'in_item_features')
in_item_features_weights = tf.placeholder("float32",[None,FEATURES_PER_POI],'in_item_features')
in_positive_items = tf.placeholder("int32",[None,FIRST_K],'in_positive_items')
in_ratings = tf.placeholder("float32",[None],'in_ratings')

initializer = tf.random_uniform_initializer(0,0.1)
#initializer = tf.glorot_uniform_initializer(seed=None)

poi_categories_np = np.array(list(poi_categories.values())).astype(np.float32)
init_poi_categories = tf.constant(poi_categories_np)

items_to_categories = tf.get_variable(name='items_to_categories',initializer=init_poi_categories)

emb_item_layer = tf.get_variable(name='item_embedding', shape=[NUM_ITEMS, ITEM_EMBEDDING_SIZE], dtype=tf.float32,
                                 initializer=initializer)

emb_feature_layer = tf.get_variable(name = 'feature_embedding', shape=[NUM_FEATURES, FEATURE_EMBEDDING_SIZE], dtype=tf.float32,
                                   initializer=initializer,trainable=TRAIN_FEATURE_EMBEDDING)

emb_user_layer = tf.get_variable(name='user_embedding', shape=[NUM_USERS,USER_EMBEDDING_SIZE], dtype=tf.float32,
                                initializer=initializer)

model_params = [emb_item_layer,emb_feature_layer,emb_user_layer]

mask_padding_zero_op = tf.scatter_update(emb_feature_layer,0,tf.zeros([FEATURE_EMBEDDING_SIZE,], dtype=tf.float32))
with tf.variable_scope("embeddings",reuse=tf.AUTO_REUSE):
    # (samples, item_embedding_size)
    emb_item = tf.nn.embedding_lookup(emb_item_layer, in_item, name = 'candidate_item_emb')
    # (samples, first_k, item_embedding_size)
    emb_positive_items = tf.nn.embedding_lookup(emb_item_layer,in_positive_items, name = 'positive_items_emb')
if (WITH_PAD):
    with tf.control_dependencies([mask_padding_zero_op]):
        # (samples, features_per_poi, feature_embedding_size)
        emb_item_features = tf.nn.embedding_lookup(emb_feature_layer, in_item_features, name = 'item_features_emb')
else:
    emb_item_features = tf.nn.embedding_lookup(emb_feature_layer, in_item_features, name = 'item_features_emb')
# (samples, features_per_poi) -> (samples,features_per_poi,1)
item_features_weights_expanded = tf.expand_dims(in_item_features_weights,-1)
# modify embedding weights
emb_item_features = tf.multiply(item_features_weights_expanded,emb_item_features)

# Learn combined item embeddings with features
multiply_item = tf.constant([1,FEATURES_PER_POI])
# (samples, item_embedding_size*features_per_poi)
dup_emb_item = tf.tile(emb_item,multiply_item,name = 'duplicated_item_emb')
# (samples, features_per_poi, item_embedding_size)
dup_emb_item = tf.reshape(dup_emb_item,[-1,FEATURES_PER_POI,ITEM_EMBEDDING_SIZE])
# (samples, features_per_poi, item_embedding_size+feature_embedding_size)
concat_emb_item = tf.concat([dup_emb_item,emb_item_features],-1,name='concat_emb_item')
# Replace weights of pad features to 0
if (WITH_PAD):
    # (samples,features_per_poi)
    feature_mask = tf.cast(tf.greater(in_item_features,0),tf.float32)
    # (samples,features_per_poi,1)
    feature_mask = tf.expand_dims(feature_mask,axis=-1)
    # (samples,features_per_poi,item_embedding_size+feature_embedding_size)
    concat_emb_item = tf.multiply(feature_mask,concat_emb_item)
# (samples, features_per_poi, item_final_embedding_size)
final_emb_item = tf.layers.dense(concat_emb_item,units=ITEM_FINAL_EMBEDDING,activation=tf.nn.relu)
dropout_emb_item = tf.nn.dropout(final_emb_item,keep_prob=DROPOUT_ITEM)

# (samples, first_k, user_embedding_size) -> [samples,item_embedding_size] * first_k
positive_items_unstacked = tf.unstack(emb_positive_items,FIRST_K,1)
if USE_RNN:
   rnn_cell = tf.contrib.rnn.GRUCell(RNN_ITEM_EMBEDDING_SIZE)
   rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=DROPOUT_RNN_USER)
   outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, positive_items_unstacked, dtype=tf.float32)
else:
   outputs = positive_items_unstacked
# (samples,rnn_item_embedding_size) * first_k -> (samples,first_k,rnn_item_embedding_size)
#contextual_positive_items = emb_positive_items
contextual_positive_items = tf.stack(outputs,axis=1)


# In[4]:


if MODEL_TYPE == 'sarah':
    # (samples, features_per_poi, item_self_attention_w)
    item_self_attention_w1 = tf.layers.dense(dropout_emb_item,units=ITEM_SELF_ATTENTION_W,activation=tf.nn.tanh)
    # (samples, features_per_poi, 1)
    item_attention_weights_raw = tf.layers.dense(item_self_attention_w1,units=1,activation=None)
    item_attention_weights = tf.nn.softmax(item_attention_weights_raw,axis=1)
    # (samples, item_final_embedding_size)
    #item_attention = emb_item
    item_attention = tf.reduce_sum(item_attention_weights*dropout_emb_item,axis=1,name='item_attention')
    # # Self attention for previous items
    # (samples, first_k, user_self_attention_w)
    user_self_attention_w1 = tf.layers.dense(contextual_positive_items, units=USER_SELF_ATTENTION_W,activation=tf.nn.tanh)
    # (samples, first_k, 1)
    user_attention_weights_raw = tf.layers.dense(user_self_attention_w1, units=1, activation=None)
    user_attention_weights = tf.nn.softmax(user_attention_weights_raw,axis=1)

    # (samples, rnn_item_embedding_size)
    user_attention = tf.reduce_sum(user_attention_weights*contextual_positive_items,axis=1,name='user_attention')
    # # Merge user and item latent vectors
    # (samples, item_final_embedding_size+rnn_item_embedding_size)
    #item_user_merged = tf.concat([item_attention,user_attention],-1,name='item_user_merged')
    # (samples, USER_ITEM_W)
    #final_emb_item = tf.layers.dense(item_user_merged,units=USER_ITEM_W,activation=tf.nn.relu)
    # (sampels,1)
    #raw_predictions_2d = tf.layers.dense(item_user_merged,units=1,activation=None)
    # (samples)
    raw_predictions = tf.reduce_sum(user_attention * item_attention,axis=1)#tf.squeeze(raw_predictions_2d)
else:
    # (samples, 1, embedding_size)
    emb_item_expanded = tf.expand_dims(emb_item,1,name='item_feature_expanded')
    # (samples, num_positives)
    attention_weights_raw = tf.reduce_sum(contextual_positive_items*emb_item_expanded, axis=-1,name='attention_weights_raw')
    attention_weights_softmax = tf.nn.softmax(attention_weights_raw, dim=-1,name='attention_weights_softmax')
    # get top r items based on attention scores (samples,r)
    top_k_vals,top_k_ind = tf.nn.top_k(attention_weights_softmax,k=TOP_ITEMS)
    # (samples, num_positives,1)
    #attention_weights_expanded = tf.expand_dims(attention_weights_softmax, -1,name='attention_weights_expanded')
    # range of samples 0,1,2,...batch size. (samples, r)
    rows = tf.tile(tf.expand_dims(tf.range(tf.shape(top_k_ind)[0]),1),[1,top_k_ind.shape[1]])
    rows_shape = tf.shape(rows)
    # (samples,r)->(samples*r) in the sort of 0,0,0,1,1,1...batch_size,batch_size
    rows_flat = tf.reshape(tf.expand_dims(rows,1),[rows_shape[0]*rows_shape[1],1])
    # (samples,r) -> (samples*r)
    top_k_ind_flat = tf.reshape(tf.expand_dims(top_k_ind,1),[rows_shape[0]*rows_shape[1],1])
    # concat rows and top indices to have list of (row,column) indices. (samples*r,2)
    ind = tf.concat([rows_flat, top_k_ind_flat],1)
    # get top r item keys (samples*r)
    res = tf.gather_nd(in_positive_items, ind)
    # return to shape by rows (samples*r)->(samples,r)
    res_final = tf.reshape(res,[tf.shape(res)[0]//rows_shape[1],rows_shape[1]])
    # Get max items' features embedding
    # Get list of features - range # features : (samples*r,features_per_poi)
    top_items_features = tf.tile(tf.transpose(tf.expand_dims(tf.range(FEATURES_PER_POI),1)),[tf.shape(res)[0],1])
    # Get their embeddings (samples*r,features_per_poi,feature_embedding_size)
    top_items_features_emb = tf.nn.embedding_lookup(emb_feature_layer, top_items_features, name = 'item_features_emb')
    # Get feature weights for items (samples*r,features_per_poi)
    top_items_weights = tf.nn.embedding_lookup(items_to_categories,res)
    # Get item embeddings (samples*r, item_embedding_size)
    top_items_embeddings = tf.nn.embedding_lookup(emb_item_layer,res)
    # (samples*r, features_per_poi) -> (samples*r,features_per_poi,1)
    top_items_weights_expanded = tf.expand_dims(top_items_weights,-1)
    # modify embedding weights (samples*r,features_per_poi,feature_embedding_size )
    top_emb_item_features = tf.multiply(top_items_weights_expanded,top_items_features_emb)
    # Learn combined item embeddings with features
    top_multiply_item = tf.constant([1,FEATURES_PER_POI])
    # (samples*r, item_embedding_size*features_per_poi)
    top_dup_emb_item = tf.tile(top_items_embeddings,top_multiply_item,name = 'top_duplicated_item_emb')
    # (samples*r, features_per_poi, item_embedding_size)
    top_dup_emb_item = tf.reshape(top_dup_emb_item,[-1,FEATURES_PER_POI,ITEM_EMBEDDING_SIZE])
    # (samples*r, features_per_poi, item_embedding_size+feature_embedding_size)
    top_concat_emb_item = tf.concat([top_dup_emb_item,top_emb_item_features],-1,name='top_concat_emb_item')
    # (samples*r, features_per_poi, item_final_embedding_size)
    top_final_emb_item = tf.layers.dense(top_concat_emb_item,units=ITEM_FINAL_EMBEDDING,activation=tf.nn.relu)
    top_dropout_emb_item = tf.nn.dropout(top_final_emb_item,keep_prob=DROPOUT_ITEM)
    # (samples*r, features_per_poi, item_final_embedding_size) -> (samples,r, features_per_poi, item_final_embedding_size)
    top_items_features_final = tf.reshape(top_dropout_emb_item,[tf.shape(res)[0]//rows_shape[1],rows_shape[1],
                                                                top_dropout_emb_item.shape[1],top_dropout_emb_item.shape[2]])
    # multiplying the candidate item feature embeddings by r (dropout_emb_item)
    # (samples, features_per_poi, item_final_embedding_size) -> (samples,r, features_per_poi, item_final_embedding_size)
    dropout_emb_item_tiled = tf.tile(tf.expand_dims(dropout_emb_item,1),[1,top_k_ind.shape[1],1,1])
    # (samples,r, features_per_poi, item_final_embedding_size) -> (samples,r,item_final_embedding_size,features_per_poi)
    top_items_features_final_t = tf.transpose(top_items_features_final,perm=[0,1,3,2])
    # Generating co-feature attention weights: 
    # (samples,r, features_per_poi, item_final_embedding_size) * (samples,r,item_final_embedding_size,features_per_poi)
    # = (samples,r, features_per_poi,features_per_poi)
    attention_top_features_raw = tf.matmul(dropout_emb_item_tiled,top_items_features_final_t)
    # softmax over two axes, taken from https://gist.github.com/raingo/a5808fe356b8da031837
    # (samples,r, features_per_poi,features_per_poi)
    with tf.name_scope('multi_softmax', 'softmax', values=[attention_top_features_raw]):
        max_axis = tf.reduce_max(attention_top_features_raw, [2,3], keep_dims=True)
        target_exp = tf.exp(attention_top_features_raw-max_axis)
        normalize = tf.reduce_sum(target_exp, [2,3], keep_dims=True)
        multi_softmax = target_exp / normalize
    # Sum all past items' weights to find candidate item most relevant feautres: (samples,r, features_per_poi)
    multi_softmax_summed = tf.reduce_sum(multi_softmax, axis=3)
    # Use attention weights to modify feature latent vectores
    # (samples,r, features_per_poi,1)
    multi_softmax_expanded = tf.expand_dims(multi_softmax_summed,3)
    # (samples,r, features_per_poi,1)*(samples,r, features_per_poi,item_final_embedding_size) - (samples,r,item_final_embedding_size)
    features_attention = tf.reduce_sum(multi_softmax_expanded*top_items_features_final,axis=2,name='features_attention')
    # Weighted average of past items based on importance
    # (samples,r) -> (samples,r,1)
    top_k_vals_expanded = tf.expand_dims(top_k_vals,2)
    # (samples,r,item_final_embedding_size)*(samples,r,1) -> (samples,item_final_embedding_size)
    features_attention_flat = tf.reduce_sum(features_attention*top_k_vals_expanded,axis=1)
    # Flatten  (samples,r,item_final_embedding_size) ->  (samples,r*item_final_embedding_size)
    #features_attention_flat = tf.reshape(features_attention,[-1,TOP_ITEMS*ITEM_FINAL_EMBEDDING])
    # Learn prediction
    raw_predictions = tf.squeeze(tf.layers.dense(features_attention_flat, units=1, activation=None))
    #attention_top_features_raw = tf.tensordot(dropout_emb_item,top_items_features_final_t,axes=[[2],[2]])
    #attention_top_features_softmax = tf.nn.softmax(attention_top_features_raw,axis=[2,3])
    #attention_top_features_raw = tf.reduce_sum(dropout_emb_item_tiled*top_items_features_final_t, axis=-1,
    #                                           name='attention_top_features_raw')
    #attention_weights_softmax = tf.nn.softmax(attention_weights_raw, dim=-1,name='attention_weights_softmax')

    # Get top features: (samples,r, features_per_poi) -> (samples,r, TOP_FEATURES)
    #top_k_features_val,top_k_features_ind = tf.nn.top_k(multi_softmax_summed,k=TOP_FEATURES)

# Transform to sigmoid
predictions = tf.sigmoid(raw_predictions)
# Loss binary cross entropy
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=in_ratings, logits=raw_predictions))
# Train step
train_step = tf.train.AdamOptimizer(LR).minimize(loss)

def get_feed_dict(dataset,start,end):
    feed_dict = {in_item : dataset['pois'][start:end],
                 in_item_features_weights : dataset['features'][start:end], 
                 in_item_features : [list(range(FEATURES_PER_POI))] * (end-start),
                 in_positive_items : dataset['k_pois'][start:end], in_ratings : dataset['scores'][start:end]}
    return feed_dict

def evaluate_model(dataset,set_size,batch_size):
    all_ratings = []
    all_predictions = []
    for i in range(0,set_size,batch_size):
        curr_ratings,curr_predictions = sess.run([in_ratings, predictions],get_feed_dict(dataset,i,i+batch_size))
        all_ratings = all_ratings + curr_ratings
        all_predictions = all_predictions + curr_predictions
    return all_ratings,all_predictions

def get_user_items(train_set,test_set):
    user_items = {}
    for i in range(len(train_set['users'])):
        if train_set['scores'][i]>=0:
            user = train_set['users'][i]
            if user not in user_items:
                user_items[user] = set()
            user_items[user].add(train_set['pois'][i])
            for p in train_set['k_pois'][i]:
                if p not in user_items[user]:
                    user_items[user].add(p)
    for i in range(len(test_set['users'])):
        if test_set['scores'][i]>=0:
            user = test_set['users'][i]
            if user not in user_items:
                user_items[user] = set()
            user_items[user].add(test_set['pois'][i])
            for p in test_set['k_pois'][i]:
                if p not in user_items[user]:
                    user_items[user].add(p)
    return user_items

def get_test_samples(test_set,poi_categories,user_items,load_path,to_load):
    test_records = []
    if (to_load == False):
        test_items = {}
        for i in range(len(test_set['scores'])):
            if test_set['scores'][i]>0:
                users_xl,pois_xl,scores_xl,k_pois_xl,features_xl = [],[],[],[],[]
                users_xl.append(test_set['users'][i])
                pois_xl.append(test_set['pois'][i])
                k_pois_xl.append(test_set['k_pois'][i])
                scores_xl.append(1)
                features_xl.append(poi_categories[test_set['pois'][i]])
                pois_curr = []
                for k in range(99):
                    j = np.random.randint(NUM_ITEMS)
                    while j in user_items[test_set['users'][i]]:
                        j = np.random.randint(NUM_ITEMS)
                    users_xl.append(test_set['users'][i])
                    pois_curr.append(j)
                    scores_xl.append(0)
                    features_xl.append(poi_categories[j])
                    k_pois_xl.append(test_set['k_pois'][i])
                test_items[i] = pois_curr
                pois_xl = pois_xl + pois_curr
                output_dict = {'pois' : pois_xl, 'users' : users_xl, 'features' : features_xl, 'k_pois' : k_pois_xl, 'scores' : scores_xl}
                test_records.append((test_set['pois'][i],output_dict))
        pickle.dump(test_items,open(load_path,'wb'))
    else:
        print("loading test items")
        test_items = pickle.load(open(load_path,'rb'))
        for i in test_items.keys():
            users_xl,pois_xl,scores_xl,k_pois_xl,features_xl = [],[],[],[],[]
            users_xl.append(test_set['users'][i])
            pois_xl.append(test_set['pois'][i])
            k_pois_xl.append(test_set['k_pois'][i])
            scores_xl.append(1)
            features_xl.append(poi_categories[test_set['pois'][i]])
            for j in test_items[i]:
                users_xl.append(test_set['users'][i])
                pois_xl.append(j)
                scores_xl.append(0)
                features_xl.append(poi_categories[j])
                k_pois_xl.append(test_set['k_pois'][i])
            output_dict = {'pois' : pois_xl, 'users' : users_xl, 'features' : features_xl, 'k_pois' : k_pois_xl, 'scores' : scores_xl}
            test_records.append((test_set['pois'][i],output_dict))
    return test_records

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

if len(sys.argv)>11:
    run_num = sys.argv[11]
else:
    run_num = ""

OUTPUT_FILE = "result/"  +PREFIX + "_" + MODEL_TYPE + "_" + str(FIRST_K) + "_" + str(self_attention_w) + "_" + str(LR) + "_" + str(DROPOUT_ITEM) + "_" + str(BATCH_SIZE) + "_" + str(NUM_FEATURES) + "_" + str(ITEM_EMBEDDING_SIZE) + "_" + str(USER_EMBEDDING_SIZE) + "_" + str(ITEM_FINAL_EMBEDDING)  +  "_" + run_num 
user_items = get_user_items(train_set,test_set)
load_path='result/test_samples_yelp_tcenr.pkl'
test_samples = get_test_samples(test_set,poi_categories,user_items,load_path,False)
config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)
sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#sess.run(tf.local_variables_initializer())
if RESTORE:
    prev_restore = tf.train.Saver()
    prev_restore.restore(sess,OUTPUT_FILE)
    print("Restored from " + OUTPUT_FILE)
else:
    best_roc_auc=0.001
    best_roc_epoch=MIN_EPOCH_TO_SAVE
    if (PRETRAIN_FILE!=None):
    #    user_emb_pre, item_emb_pre = sess.run([emb_user,emb_item],get_feed_dict(train_set,55,57))
        if USER_FINAL_EMBEDDING>0:
            pre_train_saver = tf.train.Saver({"item_embedding":emb_item_layer, "user_embedding":emb_user_layer})
        else:
            pre_train_saver = tf.train.Saver({"item_embedding":emb_item_layer})
        pre_train_saver.restore(sess,PRETRAIN_FILE)
    if (FEATURE_EMB_FILE!=None):
        pre_train_feature = pickle.load(open(FEATURE_EMB_FILE,'rb'))
        print("loading feature embedding")
        sess.run(emb_feature_layer.assign(pre_train_feature))
        print("done loading")

    saver = tf.train.Saver()

    test_indexes = list(range(0, TEST_SIZE, BATCH_SIZE))
    test_ratings=np.zeros(TEST_SIZE)
    test_output=np.zeros(TEST_SIZE)
    for start in test_indexes:
        end = min(start + BATCH_SIZE,TEST_SIZE)
        feed_dict = get_feed_dict(test_set,start,end)
        test_ratings_curr, test_output_curr = sess.run([in_ratings, predictions],feed_dict)
        test_ratings[start:end] = test_ratings_curr
        test_output[start:end] = test_output_curr
    test_accuracy = accuracy_score(y_true = test_ratings, y_pred=test_output.round())
    test_mse = mean_squared_error(y_true=test_ratings, y_pred=test_output)
    test_roc_auc = roc_auc_score(y_true=test_ratings, y_score=test_output)
    print("Test accuracy:{:.4f} mse:{:.4f} roc:{:.4f}".format(test_accuracy,test_mse,test_roc_auc))
    #test_ratings, test_output,items = sess.run([in_ratings, predictions,in_item],get_feed_dict(test_set,0,TEST_SIZE))
    #test_accuracy = accuracy_score(y_true = test_ratings, y_pred=test_output.round())
    #test_mse = mean_squared_error(y_true=test_ratings, y_pred=test_output)
    #test_roc_auc = roc_auc_score(y_true=test_ratings, y_score=test_output)
    #print("Test accuracy before training :{:.4f} mse:{:.4f} roc:{:.4f}".format(test_accuracy,test_mse,test_roc_auc))

    for epoch in range(1,EPOCHS+1):
        t1 = time()
        # Shuffle the training batch order
        training_indexes = list(range(0, TRAIN_SIZE, BATCH_SIZE))
        total_batches = len(training_indexes)
        #np.random.shuffle(training_indexes)
        train_ratings=np.zeros(TRAIN_SIZE)
        train_outputs=np.zeros(TRAIN_SIZE)
        train_losses=np.zeros(total_batches)
        curr_iter=0
        # Train the model for each batch size
        for start in training_indexes:
            end = min(start + BATCH_SIZE,TRAIN_SIZE)
            feed_dict = get_feed_dict(train_set,start,end)
            # Perform a training step for current batch
            _,curr_loss,curr_ratings, curr_output = sess.run([train_step,loss,in_ratings,predictions],feed_dict)
            train_ratings[start:end] = curr_ratings
            train_outputs[start:end] = curr_output
            train_losses[curr_iter] = curr_loss
            curr_iter+=1

        print("epoch {} took {} ms avg loss {:.4f}".format(epoch,time()-t1,np.average(train_losses)))
        train_accuracy = accuracy_score(y_true = train_ratings, y_pred=train_outputs.round())
        train_mse = mean_squared_error(y_true=train_ratings, y_pred=train_outputs)
        train_roc_auc = roc_auc_score(y_true=train_ratings, y_score=train_outputs)
        print("Train accuracy:{:.4f} mse:{:.4f} roc:{:.4f}".format(train_accuracy,train_mse,train_roc_auc))

        test_indexes = list(range(0, TEST_SIZE, BATCH_SIZE))
        test_ratings=np.zeros(TEST_SIZE)
        test_output=np.zeros(TEST_SIZE)
        for start in test_indexes:
            end = min(start + BATCH_SIZE,TEST_SIZE)
            feed_dict = get_feed_dict(test_set,start,end)
            test_ratings_curr, test_output_curr = sess.run([in_ratings, predictions],feed_dict)
            test_ratings[start:end] = test_ratings_curr
            test_output[start:end] = test_output_curr
        test_accuracy = accuracy_score(y_true = test_ratings, y_pred=test_output.round())
        test_mse = mean_squared_error(y_true=test_ratings, y_pred=test_output)
        test_roc_auc = roc_auc_score(y_true=test_ratings, y_score=test_output)
        print("Test accuracy:{:.4f} mse:{:.4f} roc:{:.4f}".format(test_accuracy,test_mse,test_roc_auc))
        
        hits, ndcgs = [],[]
        for test_record in test_samples:
            curr_poi = test_record[0]
            test_output = sess.run([predictions],get_feed_dict(test_record[1],0,100))
            map_item_score = {}
            for j in range(len(test_record[1]['pois'])):
                item = test_record[1]['pois'][j]
                map_item_score[item] = test_output[0][j]
                ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
                hr = getHitRatio(ranklist, curr_poi)
                ndcg = getNDCG(ranklist, curr_poi)
                hits.append(hr)
                ndcgs.append(ndcg)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print("hr={}, ndcg={}".format(hr,ndcg))    
        
        if (test_roc_auc/best_roc_auc)>ROC_DIFF_TO_SAVE and epoch>=MIN_EPOCH_TO_SAVE:
            save_path = saver.save(sess, OUTPUT_FILE)
            print("ROC improved from {:.4f} to {:.4f}. Model savedd to {}".format(best_roc_auc,test_roc_auc,save_path))
            best_roc_auc = test_roc_auc
            best_roc_epoch = epoch

        if (epoch - best_roc_epoch)>EARLY_STOP_INTERVAL:
            print("Early stop due to no imporvement since epoch {}".format(best_roc_epoch))
            break
sess.close()