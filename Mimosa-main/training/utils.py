import numpy as np
from sklearn.metrics import confusion_matrix
import torch


# def read_data(filepath):
#     '''read training-validation file: are site-level samples'''
#     with open(filepath) as f:
#         data = f.readlines()
#         train_positive = []
#         train_negative = []
#         val_positive = []
#         val_negative = []
#         for i in data:
#             tmp = i.strip('\n').split('\t')
#             if tmp[5] == 'train':
#                 if tmp[4] == '1':
#                     train_positive.append((tmp[1],tmp[3],[0,1])) #[0,1]
#                 elif tmp[4] == '0':
#                     train_negative.append((tmp[1],tmp[3],[1,0])) #[1,0]

#             elif tmp[5] == 'val':
#                 if tmp[4] == '1':
#                     val_positive.append((tmp[1],tmp[3],[0,1]))
#                 elif tmp[4] == '0':
#                     val_negative.append((tmp[1],tmp[3],[1,0]))

#         print('train positive',len(train_positive))
#         print('train negative',len(train_negative))
#         print('val positive',len(val_positive))
#         print('val negative',len(val_negative))
#         train = train_positive + train_negative
#         val = val_positive + val_negative
#         lens = []
#         for i in train:
#             mi,m,label = i
#             lens.append(len(mi))
#         print('min mirna len',np.min(lens))
#         return train, val

def read_data(filepath):
    '''read training-validation file: site-level samples'''
    with open(filepath) as f:
        data = f.readlines()
        train_positive = []
        train_negative = []
        val_positive = []
        val_negative = []
        for i in data:
            tmp = i.strip('\n').split('\t')
            if tmp[5] == 'train':
                if tmp[4] == '1':
                    train_positive.append((tmp[1], tmp[3], 1))  # 标签1
                elif tmp[4] == '0':
                    train_negative.append((tmp[1], tmp[3], 0))  # 标签0

            elif tmp[5] == 'val':
                if tmp[4] == '1':
                    val_positive.append((tmp[1], tmp[3], 1))  # 标签1
                elif tmp[4] == '0':
                    val_negative.append((tmp[1], tmp[3], 0))  # 标签0

        print('train positive', len(train_positive))
        print('train negative', len(train_negative))
        print('val positive', len(val_positive))
        print('val negative', len(val_negative))

        train = train_positive + train_negative
        val = val_positive + val_negative

        lens = [len(mi) for mi, m, label in train]
        print('min mirna len', np.min(lens))

        return train, val




def read_test(filepath):
    '''read test file: are gene-level samples'''
    all = []
    with open(filepath) as f:
        data = f.readlines()
        for i in data[1:]:
            tmp = i.strip('\n').split('\t')
            if tmp[4] == '1':
                all.append((tmp[1], tmp[3], 1))
            elif tmp[4] == '0':
                all.append((tmp[1], tmp[3], 0))
    return all



def reverse_seq(seq):
    '''reverse the 5-3 direction to 3-5 direction of mRNAs'''
    rseq = ''
    for i in range(len(seq)):
        rseq += seq[len(seq)-1-i]
    return rseq




def get_embedding(rna):
    '''prepared for the embedding'''
    c = {'A':0,'C':1,'G':2,'U':3,'X':4}
    map = []
    for i in range(len(rna)):
        tmp = c[rna[i]]
        map.append(tmp)
    return map


def to_Onehot(rna):
    '''Generate one-hot encoding for RNA sequence using a one-hot dictionary.'''
    # 构建一个直接映射到One-Hot编码的词典
    onehot_map = []
    onehot_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1],
        'X': [0, 0, 0, 0]  # Padding
    }
    
    # 根据词典将每个碱基映射为对应的One-Hot编码
    onehot_map = [onehot_dict[base] for base in rna]

    return np.array(onehot_map)

def to_C2(rna):
   
    C2_map = []
    C2_dict = {
        'A': [0, 0],
        'C': [1, 1],
        'G': [1, 0],
        'U': [0, 1],
        'X': [-1, -1]
    }
    C2_map = [C2_dict[base] for base in rna]
    
    return np.array(C2_map)

def to_NCP(rna):
    NCP_map = []
    NCP_dict = {
        'A': [1, 1, 1], 
        'C': [0, 1, 0], 
        'G': [1, 0, 0], 
        'U': [0, 0, 1],
        'X': [0, 0, 0]
        }
    
    NCP_map = [NCP_dict[base] for base in rna]  
    return np.array(NCP_map)

def to_ND(rna):
    ND_map = []
    base_a = base_c = base_u = base_g = base_sum = 0
    for base in rna:
        Di = 0.0
        if base == "A":
            base_a += 1
            base_sum += 1
            Di = base_a / base_sum

        elif base == "C":
                base_c += 1
                base_sum += 1
                Di = base_c / base_sum

        elif base == "U":
            base_u += 1
            base_sum += 1
            Di = base_u / base_sum
            
        elif base == "G":
            base_g += 1
            base_sum += 1
            Di = base_g / base_sum
            
        elif base == "X":
            Di = 0.0

        ND_map.append(Di)
    return np.array(ND_map)


def Smith_Waterman(seq1 ,seq2):
    '''perform sequence alignment using Smith-waterman algorithm'''
    gap = -1
    wc4 = ['AU', 'UA', 'GC', 'CG']  # Watson-Crick碱基对
    w2 = ['GU', 'UG']  # Wobble碱基对

    match_score = {'AU': 1, 'UA': 1, 'CG': 1, 'GC': 1,
                   'GU': 0, 'UG': 0, 'AC': -1, 'CA': -1,
                   'AG': -1, 'UC': -1, 'GA': -1, 'AA': -1,
                   'CC': -1, 'GG': -1, 'UU': -1, 'CU': -1,
                   'AX':-1,'XA':-1,'XC':-1,'CX':-1,
                   'GX':-1,'XG':-1,'UX':-1,'XU':-1}


    position = {'stop': 0, 'left': 1, 'up': 2, 'left_up': 3}

    m = len(seq1)
    n = len(seq2)
    score_matrix = np.zeros(( m +1, n+ 1)) # 得分矩阵初始化为0
    tracing_matrix = np.zeros((m + 1, n + 1)) # 路径追踪矩阵初始化为0


    for i in range(m + 1):
        score_matrix[i][0] = i * gap
    for j in range(n + 1):
        score_matrix[0][j] = j * gap


    max_score = -1
    max_index = (-1, -1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # print(i,j)
            match_value = match_score[seq1[i - 1] + seq2[j - 1]]
            left_up = score_matrix[i - 1, j - 1] + match_value
            up = score_matrix[i - 1, j] + gap
            left = score_matrix[i, j - 1] + gap
            score_matrix[i, j] = max(left_up, left, up, 0)

            if score_matrix[i, j] == 0:
                tracing_matrix[i, j] = position['stop']
            elif score_matrix[i, j] == left:
                tracing_matrix[i, j] = position['left']
            elif score_matrix[i, j] == up:
                tracing_matrix[i, j] = position['up']
            elif score_matrix[i, j] == left_up:
                tracing_matrix[i, j] = position['left_up']
            if score_matrix[i, j] > max_score:
                max_index = (i, j)
                max_score = score_matrix[i, j]


    align_seq1 = ''
    align_seq2 = ''
    (max_i, max_j) = max_index

    while tracing_matrix[max_i, max_j] != position['stop']:
        if tracing_matrix[max_i, max_j] == position['up']:
            align_seq1 = seq1[max_i - 1] + align_seq1
            align_seq2 = '-' + align_seq2
            max_i = max_i - 1

        elif tracing_matrix[max_i, max_j] == position['left']:
            align_seq1 = '-' + align_seq1
            align_seq2 = seq2[max_j - 1] + align_seq2
            max_j = max_j - 1

        elif tracing_matrix[max_i, max_j] == position['left_up']:
            align_seq1 = seq1[max_i - 1] + align_seq1
            align_seq2 = seq2[max_j - 1] + align_seq2
            max_i = max_i - 1
            max_j = max_j - 1

    start_index = (max_i, max_j)

    # produce pattern
    pattern = ''
    for s1, s2 in zip(align_seq1, align_seq2):
        if s1 != '-':
            if s2 != '-':
                pair = s1 + s2
                if pair in wc4:
                    pattern += '|'
                elif pair in w2:
                    pattern += ':'
                else:
                    pattern += '-'
            else:
                pattern += '-'

        else:
            if s2 != '-':
                pattern += '-'
            else:
                pattern += '-'

    # generate the pair vector of mRNA(seq2)
    pair_vector_m = [0 for p in range(len(seq2))]
    _, start_j = start_index
    gap_m = align_seq2.count('-')
    gap_count_m = 0
    for i in range(len(seq2)):
        if i < start_j:
            pair_vector_m[i] = 0
        elif (i - start_j + gap_m) < len(align_seq2):
            align_p = i - start_j
            if align_seq2[align_p] == '-':
                gap_count_m += 1
            # print(align_p+gap_count_m,align_p+gap_count_m)
            if align_seq1[align_p + gap_count_m] + align_seq2[align_p + gap_count_m] in wc4:
                pair_vector_m[i] = 1
            elif align_seq1[align_p + gap_count_m] + align_seq2[align_p + gap_count_m] in w2:
                pair_vector_m[i] = 2

    pair_vector_mi = [0 for p in range(len(seq1))]
    start_i, _ = start_index
    gap_count_mi = 0
    gap_mi = align_seq1.count('-')

    for i in range(len(seq1)):
        if i < start_i:
            pair_vector_mi[i] = 0
        elif (i - start_i + gap_mi) < len(align_seq1):
            align_p = i - start_i
            if align_seq1[align_p] == '-':
                gap_count_mi += 1
            if align_seq1[align_p + gap_count_mi] + align_seq2[align_p + gap_count_mi] in wc4:
                pair_vector_mi[i] = 1
            elif align_seq1[align_p + gap_count_mi] + align_seq2[align_p + gap_count_mi] in w2:
                pair_vector_mi[i] = 2


    return max_score, pair_vector_m, pair_vector_mi




def get_interaction_map(mirna,mrna):
    '''for training: get the pairing vector and prepare it for the embedding'''
    seed = mirna[:10]
    seed_score, seed_pair_m, seed_pair_mi = Smith_Waterman(seed,mrna[5:35])
    fseed_pair_mi = seed_pair_mi + [0] * (len(mirna) - 10)
    fseed_pair_m = [0] * 5 + seed_pair_m + [0] * 5

    return fseed_pair_m, fseed_pair_mi




def get_interaction_map_for_test(mirna,mrna):
    '''for testing samples larger than 40nt'''
    len_m = len(mrna)
    len_mi = len(mirna)

    score, m_pair, mi_pair = Smith_Waterman(mirna[:10], mrna[5:35])
    map_m = [0] * 5 + m_pair + [0] * 5
    map_mi = mi_pair + (len_mi - 10) * [0]
    return map_m, map_mi


def get_interaction_map_for_test_short(mirna,mrna):
    '''for testing samples larger than 10nt but less than 40nt'''
    len_m = len(mrna)
    len_mi = len(mirna)

    score, m_pair, mi_pair = Smith_Waterman(mirna[:10], mrna[:])
    map_m = m_pair
    map_mi = mi_pair + (len_mi - 10) * [0]
    return map_m, map_mi


def input_preprocess(seq, device):
    assert type(seq) == np.ndarray
    seq = torch.tensor(seq, dtype = torch.float32).to(device)
    return seq

def encoder(mirna, mrna, device):
    assert mirna == 26, mrna == 40
    
def decision_for_whole(data_all):
    '''define if at least one segment is predicted as functional,
    the whole mRNA sequence then will be classified as functional'''
    probabilities = [i[0] for i in data_all]
    count = len([i for i in probabilities if i > 0.5])
    if count >= 1:
        return 1
    else:
        return 0




def specificity_score(y_true, y_pred):
    '''define the evaluation metric: specificity'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity


def NPV(y_true,y_pred):
    '''define the evaluation metric: NPV'''
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    NPV = tn/(tn+fn)
    return NPV