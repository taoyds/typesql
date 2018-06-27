import re
import io
import json
import numpy as np
from lib.dbengine import DBEngine

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data

    return sql_data, table_data


def load_dataset(use_small=False):
    print "Loading from original dataset"
    sql_data, table_data = load_data('data/train_tok.jsonl',
                 'data/train_tok.tables.jsonl', use_small=use_small)
    val_sql_data, val_table_data = load_data('data/dev_tok.jsonl',
                 'data/dev_tok.tables.jsonl', use_small=use_small)

    test_sql_data, test_table_data = load_data('data/test_tok.jsonl',
                'data/test_tok.tables.jsonl', use_small=use_small)
    TRAIN_DB = 'data/train.db'
    DEV_DB = 'data/dev.db'
    TEST_DB = 'data/test.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

def best_model_name(args, for_load=False):
    new_data = 'old'
    mode = 'sqlnet'
    if for_load:
        use_emb = ''
    else:
        use_emb = '_train_emb' if args.train_emb else ''

    agg_model_name = args.sd + '/%s_%s%s.agg_model'%(new_data,
            mode, use_emb)
    sel_model_name = args.sd + '/%s_%s%s.sel_model'%(new_data,
            mode, use_emb)
    cond_model_name = args.sd + '/%s_%s%s.cond_model'%(new_data,
            mode, use_emb)

    agg_embed_name = args.sd + '/%s_%s%s.agg_embed'%(new_data, mode, use_emb)
    sel_embed_name = args.sd + '/%s_%s%s.sel_embed'%(new_data, mode, use_emb)
    cond_embed_name = args.sd + '/%s_%s%s.cond_embed'%(new_data, mode, use_emb)

    return agg_model_name, sel_model_name, cond_model_name,\
           agg_embed_name, sel_embed_name, cond_embed_name


def to_batch_seq(sql_data, table_data, idxes, st, ed, db_content=0, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []

    q_type = []
    col_type = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        if db_content == 0:
            q_seq.append([[x] for x in sql['question_tok']])
            q_type.append([[x] for x in sql["question_type_org_kgcol"]])
        else:
            q_seq.append(sql['question_tok_concol'])
            q_type.append(sql["question_type_concol_list"])
        col_type.append(table_data[sql['table_id']]['header_type_kg'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'],
            len(sql['sql']['conds']), #number of conditions + selection
            tuple(x[0] for x in sql['sql']['conds']), #col num rep in condition
            tuple(x[1] for x in sql['sql']['conds']))) #op num rep in condition, then where is str in cond?
        query_seq.append(sql['query_tok']) # real query string toks
        gt_cond_seq.append(sql['sql']['conds']) # list of conds (a list of col, op, str)
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query'], [[x] for x in sql['question_tok']]))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type


def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry, db_content):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type = \
                to_batch_seq(sql_data, table_data, perm, st, ed, db_content)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        gt_agg_seq = [x[0] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, q_type, col_type, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
        cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)


def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path, db_content):
    engine = DBEngine(db_path)
    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, db_content, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        gt_agg_seq = [x[0] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, q_type, col_type, (True, True, True))
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)

        st = ed

    return tot_acc_num / len(sql_data)


def epoch_acc(model, batch_size, sql_data, table_data, pred_entry, db_content, error_print=False):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type,\
         raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, db_content, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, q_type, col_type, pred_entry)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry, error_print)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)


def load_para_wemb(file_name):
    f = io.open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    ret = {}
    if len(lines[0].split()) == 2:
        lines.pop(0)
    for (n,line) in enumerate(lines):
        info = line.strip().split(' ')
        if info[0].lower() not in ret:
            ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))

    return ret


def load_comb_wemb(fn1, fn2):
    wemb1 = load_word_emb(fn1)
    wemb2 = load_para_wemb(fn2)
    comb_emb = {k: wemb1.get(k, 0) + wemb2.get(k, 0) for k in set(wemb1) | set(wemb2)}

    return comb_emb


def load_concat_wemb(fn1, fn2):
    wemb1 = load_word_emb(fn1)
    wemb2 = load_para_wemb(fn2)
    backup = np.zeros(300, dtype=np.float32)
    comb_emb = {k: np.concatenate((wemb1.get(k, backup), wemb2.get(k, backup)), axis=0) for k in set(wemb1) | set(wemb2)}

    return None, None, comb_emb


def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print ('Loading word embedding from %s'%file_name)
        ret = {}
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))
        return ret
    else:
        print ('Load used word embedding')
        with open('glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val


def load_word_and_type_emb(fn1, fn2, sql_data, table_data, db_content, is_list=False, use_htype=False):
    word_to_idx = {'<UNK>':0, '<BEG>':1, '<END>':2}
    word_num = 3
    N_word = 300
    embs = [np.zeros(N_word, dtype=np.float32) for _ in range(word_num)]
    _, _, word_emb = load_concat_wemb(fn1, fn2)

    if is_list:
        for sql in sql_data:
            if db_content == 0:
                qtype = [[x] for x in sql["question_type_org_kgcol"]]
            else:
                qtype = sql['question_type_concol_list']
            for tok_typl in qtype:
                tys = " ".join(sorted(tok_typl))
                if tys not in word_to_idx:
                    emb_list = []
                    ws_len = len(tok_typl)
                    for w in tok_typl:
                        if w in word_emb:
                            emb_list.append(word_emb[w][:N_word])
                        else:
                            emb_list.append(np.zeros(N_word, dtype=np.float32))
                    word_to_idx[tys] = word_num
                    word_num += 1
                    embs.append(sum(emb_list) / float(ws_len))

        if use_htype:
            for tab in table_data.values():
                for col in tab['header_type_kg']:
                    cts = " ".join(sorted(col))
                    if cts not in word_to_idx:
                        emb_list = []
                        ws_len = len(col)
                        for w in col:
                            if w in word_emb:
                                emb_list.append(word_emb[w][:N_word])
                            else:
                                emb_list.append(np.zeros(N_word, dtype=np.float32))
                        word_to_idx[cts] = word_num
                        word_num += 1
                        embs.append(sum(emb_list) / float(ws_len))

    else:
        for sql in sql_data:
            if db_content == 0:
                qtype = sql['question_tok_type']
            else:
                qtype = sql['question_type_concol_list']
            for tok in qtype:
                if tok not in word_to_idx:
                    word_to_idx[tok] = word_num
                    word_num += 1
                    embs.append(word_emb[tok][:N_word])

        if use_htype:
            for tab in table_data.values():
                for tok in tab['header_type_kg']:
                    if tok not in word_to_idx:
                        word_to_idx[tok] = word_num
                        word_num += 1
                        embs.append(word_emb[tok][:N_word])


    agg_ops = ['null', 'maximum', 'minimum', 'count', 'total', 'average']
    for tok in agg_ops:
        if tok not in word_to_idx:
            word_to_idx[tok] = word_num
            word_num += 1
            embs.append(word_emb[tok][:N_word])

    emb_array = np.stack(embs, axis=0)

    return (word_to_idx, emb_array, word_emb)
