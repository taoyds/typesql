
# coding: utf-8

# In[1]:


import json
import os
import sys
import argparse

# In[2]:


class Keys(object):
    # keys for question
    KG_ENTITIES = "kg_entities"
    QUESTION = "question"
    QUESTION_TOK_KGCOL = "question_tok_kgcol"
    QUESTION_TYPE_KGCOL = "question_type_kgcol"
    QUESTION_TYPE_KGCOL_LIST = "question_type_kgcol_list"
    QUESTION_TYPE_ORG_KGCOL = "question_type_org_kgcol"
    QUESTION_TOK_ORG = "question_tok_org"
    QUESTION_TABLE_ID = "table_id"
    QUESTION_TOK_CONCOL = "question_tok_concol"
    QUESTION_TYPE_CONCOL = "question_type_concol"
    QUESTION_TYPE_CONCOL_LIST = "question_type_concol_list"
    QUESTION_TYPE_ORG_CONCOL = "question_type_org_concol"
    QUESTION_TOK_SPACE = "question_tok_space"
    
    # keys for table
    TABLE_ID = "id"
    TABLE_ROWS = "rows"
    HEADER_TOK = "header_tok"
    HEADER_TYPE = "header_type_kg"
    
    # old keys
    QUESTION_TOK_TYPE = "question_tok_type" # => q_type_kgcol_list
    QUESTION_TOK = "question_tok" # => q_tok_kgcol
    
    # keys for meta information
    # meta header: {TYPE: TYPE_HEADER, META_CLS: ..., META_SIM_CLS: ...}
    # meta type: {TYPE: TYPE_TYPE, META_CLS: ..., META_SIM_CLS: ...}
    # meta kg: {TYPE: TYPE_KG, META_CLS: ..., META_SIM_CLS: ...}
    META = "meta"
    TYPE = "meta_type"
    TYPE_HEADER = "header"
    TYPE_TYPE = "type"
    TYPE_KG = "kg"
    TYPE_NONE = "none"
    META_CLS = "cls"
    META_SIM_CLS = "sim_cls"
    META_TOKS = "meta_tok"
    
    # keys for meta class
    META_DATE = "date"
    META_YEAR = "year"
    META_GAME_SCORE = "game score"
    META_INT = "integer"
    META_INT_SMALL = "small integer"
    META_INT_NEG = "negative integer"
    META_INT_MED = "medium integer"
    META_INT_BIG = "big integer"
    META_INT_LARGE = "large integer"
    META_FLOAT = "float"
    META_FLOAT_SMALL = "small float"
    META_FLOAT_NEG = "negative float"
    META_FLOAT_MED = "medium float"
    META_FLOAT_LARGE = "large float"
    
    META_LIST = ["person", "country", "place", "organization", "sport"]
    META_MAP = {
        META_DATE: META_DATE,
        META_YEAR: META_YEAR,
        META_GAME_SCORE: META_INT,
        META_INT_SMALL: META_INT,
        META_INT_NEG: META_INT,
        META_INT_MED: META_INT,
        META_INT_BIG: META_INT,
        META_INT_LARGE: META_INT,
        META_INT: META_INT,
        META_FLOAT: META_FLOAT,
        META_FLOAT_SMALL: META_FLOAT,
        META_FLOAT_NEG: META_FLOAT,
        META_FLOAT_MED: META_FLOAT,
        META_FLOAT_LARGE: META_FLOAT
    }
    META_MAP["person"] = "person"
    META_MAP["country"] = "country"
    META_MAP["place"] = "place"
    META_MAP["organization"] = "organization"
    META_MAP["sport"] = "sportsteam"
    
    # randomly chosen NONE string
    NONE = "te8r2ed" 
    COLUMN = "column"
    ENTITY = "entity"


# In[3]:


def get_date(_in):
    """
    string to date
    """
    month_lst = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'Jul',
              'aug', 'sep', 'oct', 'nov', 'dec']
    
    def ismonth(tok):
        for mon in month_lst:
            if mon in tok: return True
        return False
    
    has_num = False
    has_m = False
    sps = _in.lower().split(" ")
    for s in sps:
        if s.isdigit():
            has_num = True
        else:
            has_m = ismonth(s)
        if has_m and has_num and (sps[0].isdigit() or ismonth(sps[0])):
            return _in, Keys.META_DATE.split(), Keys.META_DATE
            
    return None
    
def get_float(_in):
    """
    string to float
    """
    try:
        num = float(_in)
        cls = Keys.META_FLOAT
        if num < 100: 
            cls = Keys.META_FLOAT_SMALL
        elif 100 <= num < 1000:
            cls = Keys.META_FLOAT_MED
        elif 1000 <= num:
            cls = Keys.META_FLOAT_LARGE
        sim_cls = Keys.META_MAP[cls]
    except ValueError:
        return None
    return num, cls.split(), sim_cls

def get_int(_in):
    """
    string to int
    """
    try:
        num = int(_in)
        cls = Keys.META_INT
        if num < 100: 
            cls = Keys.META_INT_SMALL
        elif 100 <= num < 1000:
            cls = Keys.META_INT_MED
        elif 1000 <= num:
            cls = Keys.META_INT_LARGE
        sim_cls = Keys.META_MAP[cls]
    except ValueError:
        return None
    return num, cls.split(), sim_cls

def get_score(_in):
    """
    string to score
    """
    if '-' in _in:
        try:
            score = int(_in.split("-")[0])
            if score < 50: 
                cls = Keys.META_GAME_SCORE
                sim_cls = Keys.META_MAP[cls]
                return score, cls.split(), sim_cls
            else: return None
        except ValueError:
            return None
    return None

def get_year(_in):
    """
    string to year
    """
    if '-' in _in:
        try:
            year = int(_in.split("-")[0])
            if 1200 < year < 2100: 
                cls = Keys.META_YEAR
                sim_cls = Keys.META_MAP[cls]
                return year, cls.split(), sim_cls
            else: return None
        except ValueError:
            return None
    return None

def get_header(tokens, idx, num_toks, header_tok, header_type):
    """
    list to header, return end idx, header_class
    """
    for endIdx in reversed(range(idx+1, num_toks+1)):
        sub_toks = tokens[idx: endIdx]
        if sub_toks in header_tok:
            cls = header_type[header_tok.index(sub_toks)]
            k = " ".join(cls)
            sim_cls = Keys.COLUMN 
            return endIdx, cls, sim_cls
    return None

def get_kg(tokens, idx, num_toks, kg_tok, kg_type):
    """
    list to header, return end idx, kg_class
    """
    for endIdx in reversed(range(idx+1, num_toks+1)):
        sub_toks = tokens[idx: endIdx]
        if sub_toks in kg_tok:
            cls = kg_type[kg_tok.index(sub_toks)]
            sim_cls = Keys.NONE
            for k in Keys.META_LIST:
                if k in cls:
                    sim_cls = Keys.META_MAP[k]
                    break
            cls = [sim_cls]
            return endIdx, cls, sim_cls
    return None


# In[4]:


def group_words(entry, tables):
    """
    Group words in order of header, type, KG
    and add to entry's meta
    """
    # initialize entry's meta data
    entry[Keys.META] = list()
    kg_tok = [sublist[0] for sublist in entry[Keys.KG_ENTITIES]]
    kg_type = [sublist[1] for sublist in entry[Keys.KG_ENTITIES]]
    tokens = entry[Keys.QUESTION_TOK_ORG]
    table = tables[entry[Keys.QUESTION_TABLE_ID]]
    header_tok = table[Keys.HEADER_TOK]
    header_type = table[Keys.HEADER_TYPE]
    
    num_toks = len(tokens)
    idx = 0
    
    def build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx):
        this_entry[Keys.META_CLS] = cls
        this_entry[Keys.META_SIM_CLS] = sim_cls
        this_entry[Keys.META_TOKS] = tokens[idx: endIdx]
        return this_entry
    
    while idx < num_toks:
        this_entry = dict()
        
        res = get_header(tokens, idx, num_toks, header_tok, header_type)
        if res:
            this_entry[Keys.TYPE] = Keys.TYPE_HEADER
            endIdx, cls, sim_cls = res
            entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
            idx = endIdx
            continue
            
        res = get_score(tokens[idx])
        if res:
            this_entry[Keys.TYPE] = Keys.TYPE_TYPE
            _, cls, sim_cls = res
            endIdx = idx + 1
            entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
            idx = endIdx
            continue
        
        res = get_year(tokens[idx])
        if res:
            this_entry[Keys.TYPE] = Keys.TYPE_TYPE
            _, cls, sim_cls = res
            endIdx = idx + 1
            entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
            idx = endIdx
            continue
            
        if idx + 2 <= num_toks:
            this_entry[Keys.TYPE] = Keys.TYPE_TYPE
            res = get_date(" ".join(tokens[idx:idx+2]))
            if res:
                _, cls, sim_cls = res
                endIdx = idx + 2
                entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
                idx = endIdx
                continue
            
        if idx + 3 <= num_toks:
            this_entry[Keys.TYPE] = Keys.TYPE_TYPE
            res = get_date(" ".join(tokens[idx:idx+3]))
            if res:
                _, cls, sim_cls = res
                endIdx = idx + 3
                entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
                idx = endIdx
                continue
            
        res = get_int(tokens[idx])
        if res:
            this_entry[Keys.TYPE] = Keys.TYPE_TYPE
            _, cls, sim_cls = res
            endIdx = idx + 1
            entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
            idx = endIdx
            continue
            
        res = get_float(tokens[idx])
        if res:
            this_entry[Keys.TYPE] = Keys.TYPE_TYPE
            _, cls, sim_cls = res
            endIdx = idx + 1
            entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
            idx = endIdx
            continue
            
        res = get_kg(tokens, idx, num_toks, kg_tok, kg_type)
        if res:
            this_entry[Keys.TYPE] = Keys.TYPE_KG
            endIdx, cls, sim_cls = res
            entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
            idx = endIdx
            continue
        
        this_entry[Keys.TYPE] = Keys.TYPE_NONE
        cls = [Keys.NONE]
        sim_cls = Keys.NONE
        endIdx = idx + 1
        entry[Keys.META].append(build_entry(this_entry, tokens, cls, sim_cls, idx, endIdx))
        idx = endIdx
        
    return entry


# In[5]:


def get_header_col(tokens, idx, num_toks, header_tok):
    for endIdx in reversed(range(idx+1, num_toks+1)):
        sub_toks = tokens[idx: endIdx]
        if sub_toks in header_tok:
            return endIdx, sub_toks
    return None

def get_con_col(tokens, token_space, idx, num_toks, header_tok, rows, cols):
    find = []
    for endIdx in reversed(range(idx+1, num_toks+1)):
        sub_toks = tokens[idx: endIdx]
        tok_space = token_space[idx: endIdx]
        cont = ""
        for tok, space in zip(sub_toks, tok_space):
            cont += tok + space
        cont = cont.strip()
        for r in rows:
            for i, c in enumerate(r):
                try:
                    if cont == str(c):
                        find.append((endIdx, header_tok[i]))
                        continue
                except:
                    pass
                
                try:
                    c = c.lower()
                except:
                    continue
                if cont == c: 
                    find.append((endIdx, header_tok[i]))
        if len(find) > 0: break
    if len(find) > 0:
        for f in find:
            if f[1] in cols:
                return f
        return find[0]
    return None


# In[6]:


def group_words_col(entry, tables):
    """
    Group words in order of header, content
    and add to entry's meta
    """
    entry[Keys.QUESTION_TYPE_CONCOL_LIST] = list()
    entry[Keys.QUESTION_TYPE_CONCOL] = list()
    entry[Keys.QUESTION_TYPE_ORG_CONCOL] = list()
    entry[Keys.QUESTION_TOK_CONCOL] = list()
    tokens = entry[Keys.QUESTION_TOK_ORG]
    token_space = entry[Keys.QUESTION_TOK_SPACE]
    table = tables[entry[Keys.QUESTION_TABLE_ID]]
    header_tok = table[Keys.HEADER_TOK]
    rows = table[Keys.TABLE_ROWS]
    
    num_toks = len(tokens)
    idx = 0
    
    cols = list()
    
    while idx < num_toks:
        res = get_header_col(tokens, idx, num_toks, header_tok)
        if res:
            endIdx, _ = res
            entry[Keys.QUESTION_TYPE_CONCOL_LIST].append([Keys.COLUMN])
            entry[Keys.QUESTION_TYPE_CONCOL].append(Keys.COLUMN)
            entry[Keys.QUESTION_TYPE_ORG_CONCOL] += [Keys.COLUMN] * (endIdx - idx)
            entry[Keys.QUESTION_TOK_CONCOL].append(tokens[idx: endIdx])
            cols.append(tokens[idx: endIdx])
            idx = endIdx
            continue
        
        res = get_con_col(tokens, token_space, idx, num_toks, header_tok, rows, cols)
        if res:
            endIdx, col_name = res
#             print 'col_name', col_name
            entry[Keys.QUESTION_TYPE_CONCOL_LIST].append(col_name)
            entry[Keys.QUESTION_TYPE_CONCOL].append(Keys.ENTITY)
            entry[Keys.QUESTION_TYPE_ORG_CONCOL] += [Keys.ENTITY] * (endIdx - idx)
            entry[Keys.QUESTION_TOK_CONCOL].append(tokens[idx: endIdx])
            idx = endIdx
            continue
        
        entry[Keys.QUESTION_TYPE_CONCOL_LIST].append([Keys.NONE])
        entry[Keys.QUESTION_TYPE_CONCOL].append(Keys.NONE)
        entry[Keys.QUESTION_TYPE_ORG_CONCOL].append(Keys.NONE)
        entry[Keys.QUESTION_TOK_CONCOL].append([tokens[idx]])
        idx += 1
    
    return entry


# In[7]:


def add_question_tok_kgcol(entry):
    kg_entities = entry[Keys.KG_ENTITIES]
    toks = entry[Keys.QUESTION_TOK]
    
    idx = 0
    num_toks = len(toks)
    tok_kgcol = list()
    while idx < num_toks:
        e = list()
        for endIdx in reversed(range(idx+1, num_toks+1)):
            sub_toks = toks[idx: endIdx]
            if sub_toks in kg_entities:
                e = sub_toks
                idx = endIdx
                break
        if len(e) == 0:
            e.append(toks[idx])
            idx += 1
        tok_kgcol.append(e)
            
    entry[Keys.QUESTION_TOK_KGCOL] = tok_kgcol
    
    return entry


# In[8]:


def load_and_process_data(file_path, table_path, out_path):
    data = list()
    tables = dict()
    with open(file_path) as f:
        for line in f:
            data = [json.loads(line.strip()) for line in f]
    
    with open(table_path) as f:
        for line in f:
            table = json.loads(line.strip())
            tables[table[Keys.TABLE_ID]] = table
    print len(data)
    with open(out_path, 'w') as f:
        for idx, entry in enumerate(data):
            # change from old keys to new keys
#             if Keys.QUESTION_TOK_TYPE in entry:
#                 entry[Keys.QUESTION_TYPE_KGCOL_LIST] = entry[Keys.QUESTION_TOK_TYPE]
#                 del entry[Keys.QUESTION_TOK_TYPE]
#             if Keys.QUESTION_TOK in entry:
#                 entry[Keys.QUESTION_TOK_KGCOL] = entry[Keys.QUESTION_TOK]
#                 entry[Keys.QUESTION_TOK_ORG] = [item for sublist in entry[Keys.QUESTION_TOK] for item in sublist]
#                 entry[Keys.QUESTION_TOK] = entry[Keys.QUESTION_TOK_ORG]
            entry[Keys.QUESTION_TOK_ORG] = entry[Keys.QUESTION_TOK]
            # add question_tok_kgcol
            entry = add_question_tok_kgcol(entry)
        
            entry = group_words(entry, tables)
            
            # add question_tok_kgcol
            res = [item[Keys.META_TOKS] for item in entry[Keys.META]]
          
            entry[Keys.QUESTION_TOK_KGCOL] = res
            
            # add question_type_kgcol
            res = [item[Keys.META_SIM_CLS] for item in entry[Keys.META]]
            entry[Keys.QUESTION_TYPE_KGCOL] = res
            
            # add question_type_kgcol_list
            res = []
            for item in entry[Keys.META]:
                extra = []
                if item[Keys.TYPE] == Keys.TYPE_KG: extra.append(Keys.ENTITY)
                elif item[Keys.TYPE] == Keys.TYPE_HEADER: extra.append(Keys.COLUMN)
                try:
                    res += [item[Keys.META_CLS] + extra]
                except:
                    print item
                    print entry
                    print '------------------------------------'
            entry[Keys.QUESTION_TYPE_KGCOL_LIST] = res
            
            # add question_type_org_kgcol
            res = []
            for item in entry[Keys.META]:
                res += [item[Keys.META_SIM_CLS]] * len(item[Keys.META_TOKS])
            entry[Keys.QUESTION_TYPE_ORG_KGCOL] = res
            del entry[Keys.META]
            if (idx + 1) % 100 == 0:
                print idx + 1
            entry = group_words_col(entry, tables)
            f.write(json.dumps(entry) + "\n")


# In[9]:

def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='process data')
    parser.add_argument('--tok', dest='tok',
                        default=None, type=str)
    parser.add_argument('--table', dest='table',
                        default=None, type=str)
    parser.add_argument('--out', dest='out',
                        default=None, type=str)
    parser.add_argument('--data_dir', dest='data_dir',
                        default='data', type=str)
    parser.add_argument('--out_dir', dest='out_dir',
                        default='tmp', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    file_path = os.path.join(args.data_dir, args.tok)
    table_path = os.path.join(args.data_dir, args.table)
    out_path = os.path.join(args.out_dir, args.out)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    load_and_process_data(file_path, table_path, out_path)
