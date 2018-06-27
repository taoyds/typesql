import json
import torch
import datetime
import argparse
import numpy as np
from typesql.utils import *
from typesql.model.sqlnet import SQLNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--sd', type=str, default='',
            help='set model save directory.')
    parser.add_argument('--db_content', type=int, default=0,
            help='0: use knowledge graph type, 1: use db content to get type info')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet.')
    args = parser.parse_args()

    N_word=600
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=True
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=True
        BATCH_SIZE=64
    TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)

    #word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
    #        load_used=args.train_emb, use_small=USE_SMALL)
    if args.db_content == 0:
        word_emb = load_word_and_type_emb('glove/glove.42B.300d.txt', "para-nmt-50m/data/paragram_sl999_czeng.txt",\
                                         val_sql_data, val_table_data, args.db_content, is_list=True, use_htype=False)
    else:
        word_emb = load_concat_wemb('glove/glove.42B.300d.txt', "para-nmt-50m/data/paragram_sl999_czeng.txt")

    model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb = args.train_emb, db_content=args.db_content)

    agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)
    print "Loading from %s"%agg_m
    model.agg_pred.load_state_dict(torch.load(agg_m))
    print "Loading from %s"%sel_m
    model.selcond_pred.load_state_dict(torch.load(sel_m))
    print "Loading from %s"%cond_m
    model.op_str_pred.load_state_dict(torch.load(cond_m))
    #only for loading trainable embedding
    print "Loading from %s"%agg_e
    model.agg_type_embed_layer.load_state_dict(torch.load(agg_e))
    print "Loading from %s"%sel_e
    model.sel_type_embed_layer.load_state_dict(torch.load(sel_e))
    print "Loading from %s"%cond_e
    model.cond_type_embed_layer.load_state_dict(torch.load(cond_e))


    print "Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY, args.db_content)
    print "Dev execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB, args.db_content)
    print "Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY, args.db_content)
    print "Test execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, args.db_content)
